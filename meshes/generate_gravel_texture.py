#!/usr/bin/env python3
"""
Procedural basaltic gravel texture generator for the LunaLab digital twin.

Physical target: 0.2-4mm grains (per SPICE-HL3 paper), tiled at 2m per
texture repeat (matching <size>2</size> in lunalab.sdf's heightmap texture).
At 4096x4096px over a 2m tile, that's ~0.49mm/pixel -- enough resolution
to actually depict individual grains, unlike the old 512x512 (3.9mm/px)
texture which was mathematically incapable of showing grain-scale detail.

Approach:
  1. Build a height field by stamping many small rounded "grain" bumps
     at random positions/sizes, using periodic (toroidal) distance so
     the result tiles seamlessly with no visible seam when repeated.
  2. Derive a diffuse (color) map from the same height field, with
     per-grain basaltic gray/brown color variation plus height-based
     shading for readability.
  3. Derive a normal map from the height field's gradient, so the
     lighting detail is physically consistent with the same bumps.
"""

import numpy as np
from PIL import Image

# ---- Parameters ----
RES = 4096                    # texture resolution (pixels per tile edge)
TILE_SIZE_MM = 2000.0          # matches <size>2</size> in the SDF (2m = 2000mm)
PX_PER_MM = RES / TILE_SIZE_MM

GRAIN_MIN_MM, GRAIN_MAX_MM = 0.2, 4.0   # paper's stated LunaLab grain range
N_GRAINS = 9000                          # grain count (density tradeoff vs runtime)

rng = np.random.default_rng(42)  # fixed seed = reproducible output

def toroidal_delta(a, b, size):
    """Shortest signed distance between two coordinates on a wrapping (tileable) axis."""
    d = a - b
    return d - size * np.round(d / size)

# ---- 1. Height field: stamp grains with periodic wraparound ----
height = np.zeros((RES, RES), dtype=np.float32)
grain_id = np.full((RES, RES), -1, dtype=np.int32)   # which grain "owns" each pixel (for coloring)

yy, xx = np.mgrid[0:RES, 0:RES].astype(np.float32)

print(f"Stamping {N_GRAINS} grains onto a {RES}x{RES} height field...")
for i in range(N_GRAINS):
    cx = rng.uniform(0, RES)
    cy = rng.uniform(0, RES)
    r_mm = rng.uniform(GRAIN_MIN_MM, GRAIN_MAX_MM)
    r_px = max(r_mm * PX_PER_MM, 0.6)   # floor so the smallest grains still register >=1px
    grain_height = rng.uniform(0.4, 1.0) * r_px   # bigger grains sit a bit taller, roughly

    # Only touch a local bounding box around the grain, not the whole image --
    # this is what keeps a 9000-grain loop fast instead of astronomically slow.
    pad = int(np.ceil(r_px)) + 1
    x0, x1 = int(cx) - pad, int(cx) + pad
    y0, y1 = int(cy) - pad, int(cy) + pad

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    gx, gy = np.meshgrid(xs, ys)

    dx = toroidal_delta(gx.astype(np.float32), cx, RES)
    dy = toroidal_delta(gy.astype(np.float32), cy, RES)
    dist = np.sqrt(dx * dx + dy * dy)

    mask = dist < r_px
    # Smooth rounded bump profile (cosine falloff), not a hard-edged disc --
    # real grains don't have razor-sharp silhouettes.
    bump = np.where(mask, grain_height * (np.cos(np.pi / 2 * dist / r_px) ** 2), 0.0)

    gxi = gx[mask] % RES
    gyi = gy[mask] % RES
    bump_vals = bump[mask]

    cur = height[gyi, gxi]
    take = bump_vals > cur   # "max blend": grains stack/overlap like real piled gravel,
    height[gyi, gxi] = np.where(take, bump_vals, cur)  # rather than adding heights together
    grain_id[gyi, gxi] = np.where(take, i, grain_id[gyi, gxi])

    if (i + 1) % 1500 == 0:
        print(f"  {i+1}/{N_GRAINS}")

# Fine sub-grain roughness so flat gaps between grains aren't perfectly smooth
fine_noise = rng.normal(0, 0.15, size=(RES, RES)).astype(np.float32)
# cheap 3x3 box blur to keep the fine noise from looking like pure static
from numpy.lib.stride_tricks import sliding_window_view
padded = np.pad(fine_noise, 1, mode="wrap")
fine_noise = sliding_window_view(padded, (3, 3)).mean(axis=(-1, -2))
height += fine_noise

print("Height field complete. Building diffuse + normal maps...")

# ---- 2. Diffuse (color) map ----
# Basaltic gravel: dark gray base with some grains leaning warmer/browner
# (iron staining), matching the existing material's ambient/diffuse tone
# already in lunalab.sdf (~0.30 0.27 0.22).
# NOTE: first attempt at these values was too dark -- combined with the
# paper's deliberately grazing-angle sunlight (4-8 deg elevation means very
# little direct light hits flat ground), the terrain rendered as almost
# invisible. Brightened here; basalt is genuinely dark rock, but not this dark.
base_gray = rng.uniform(0.38, 0.56, size=N_GRAINS).astype(np.float32)
warm_shift = rng.uniform(-0.03, 0.07, size=N_GRAINS).astype(np.float32)
grain_r = base_gray + warm_shift * 0.6
grain_g = base_gray + warm_shift * 0.35
grain_b = base_gray - warm_shift * 0.4

# Background (gaps between grains) - slightly darker, matches shadowed crevices
bg_color = np.array([0.30, 0.27, 0.23], dtype=np.float32)

diffuse = np.tile(bg_color, (RES, RES, 1)).astype(np.float32)
has_grain = grain_id >= 0
gid = grain_id[has_grain]
diffuse[has_grain, 0] = grain_r[gid]
diffuse[has_grain, 1] = grain_g[gid]
diffuse[has_grain, 2] = grain_b[gid]

# Height-based shading: higher points catch a bit more light, low points
# (crevices between grains) read darker -- cheap pseudo-ambient-occlusion
# baked directly into the diffuse map, which reads well even before the
# real directional light + normal map add further shading at render time.
h_norm = (height - height.min()) / (np.ptp(height) + 1e-6)
shade = 0.85 + 0.30 * h_norm  # keep shading subtle, not blown out
diffuse *= shade[..., None]
diffuse = np.clip(diffuse, 0, 1)

diffuse_img = Image.fromarray((diffuse * 255).astype(np.uint8), mode="RGB")

# ---- 3. Normal map, derived from the height field's gradient ----
# Standard tangent-space normal map convention: encode the surface normal
# vector (nx, ny, nz) into RGB as (nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5).
# A perfectly flat surface -> normal (0,0,1) -> RGB (128,128,255), which
# is why normal maps look flat blue-purple at a glance.
gy_grad, gx_grad = np.gradient(height)
strength = 2.2  # exaggerate slightly so grain bumps read clearly at render distance
nx = -gx_grad * strength
ny = -gy_grad * strength
nz = np.ones_like(height)
norm_len = np.sqrt(nx**2 + ny**2 + nz**2)
nx, ny, nz = nx / norm_len, ny / norm_len, nz / norm_len

normal_rgb = np.stack([
    (nx * 0.5 + 0.5),
    (ny * 0.5 + 0.5),
    (nz * 0.5 + 0.5),
], axis=-1)
normal_img = Image.fromarray((np.clip(normal_rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")

diffuse_img.save("/home/claude/lunalab_diffuse.png")
normal_img.save("/home/claude/lunalab_normal.png")
print("Saved lunalab_diffuse.png and lunalab_normal.png")
