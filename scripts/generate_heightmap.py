import numpy as np
from PIL import Image

RES = 257          # standard heightmap resolution (2^n + 1)
WORLD_X = 11.0      # meters, matches world SDF size
WORLD_Y = 8.0
MAX_ELEVATION_RANGE = 0.4   # meters spanned from black(0) to white(255)
GROUND_GRAY = 128           # this gray value = "ground level" z=0

rng = np.random.default_rng(42)
heightmap = np.full((RES, RES), float(GROUND_GRAY))

# Subtle gravel-like noise, lightly smoothed (no scipy dependency)
# std=6 gives visible fine texture without looking exaggerated (~9mm-scale variation)
noise = rng.normal(0, 6.0, (RES, RES))
for _ in range(3):
    noise = (np.roll(noise, 1, 0) + np.roll(noise, -1, 0) +
             np.roll(noise, 1, 1) + np.roll(noise, -1, 1) + noise) / 5.0
heightmap += noise

def world_to_pixel(wx, wy):
    px = (wx + WORLD_X / 2) / WORLD_X * (RES - 1)
    py = (wy + WORLD_Y / 2) / WORLD_Y * (RES - 1)
    return px, py

def add_crater(wx, wy, radius_m, depth_gray):
    px, py = world_to_pixel(wx, wy)
    # FIX: previously both axes used WORLD_X for pixel-per-meter scaling, which
    # distorted craters into ellipses since the heightmap is square (257x257)
    # but the room isn't (11m x 8m). Now each axis uses its own scale.
    radius_px_x = radius_m / WORLD_X * (RES - 1)
    radius_px_y = radius_m / WORLD_Y * (RES - 1)
    yy, xx = np.mgrid[0:RES, 0:RES]
    dist_sq = ((xx - px) / radius_px_x) ** 2 + ((yy - py) / radius_px_y) ** 2
    return depth_gray * np.exp(-dist_sq / (2 * (0.5) ** 2))

# Craters: (world_x, world_y, radius_m, depth in gray-value units)
# depth 60-100 gray -> roughly 9-16cm deep, a realistic scale for small lab craters
craters = [
    (0.5, 2.5, 0.6, 70),
    (-2.5, -1.0, 0.5, 60),
    (-0.8, -3.0, 0.4, 55),
    (3.5, 0.0, 0.45, 65),
    (-3.5, 1.0, 0.35, 50),
    (1.5, -2.5, 0.55, 75),
]
for wx, wy, r, depth in craters:
    heightmap -= add_crater(wx, wy, r, depth)

heightmap = np.clip(heightmap, 0, 255).astype(np.uint8)
Image.fromarray(heightmap, mode='L').save('lunalab_heightmap.png')
print("Saved lunalab_heightmap.png")
print(f"Ground-level pos.z offset to use in SDF: {-(GROUND_GRAY/255)*MAX_ELEVATION_RANGE:.4f}")

# --- Minimal diffuse + normal textures (required by Ogre2 heightmap shader) ---
TEX_RES = 512
diffuse_base = np.array([90, 80, 68])  # RGB, dusty regolith tone
diffuse_noise = rng.integers(-15, 15, (TEX_RES, TEX_RES, 3))
diffuse = np.clip(diffuse_base + diffuse_noise, 0, 255).astype(np.uint8)
Image.fromarray(diffuse, mode='RGB').save('lunalab_diffuse.png')

normal = np.full((TEX_RES, TEX_RES, 3), [128, 128, 255], dtype=np.uint8)
Image.fromarray(normal, mode='RGB').save('lunalab_normal.png')
print("Saved lunalab_diffuse.png and lunalab_normal.png")
