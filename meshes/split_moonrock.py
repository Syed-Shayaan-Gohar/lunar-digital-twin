#!/usr/bin/env python3
import re
import os

SRC = "moonrock.obj"
OUTDIR = "rocks_split"
os.makedirs(OUTDIR, exist_ok=True)

verts, uvs, norms = [], [], []
groups = {}  # name -> list of face lines (raw, as v/vt/vn strings)
current = None

with open(SRC) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith("v "):
            verts.append(line)
        elif line.startswith("vt "):
            uvs.append(line)
        elif line.startswith("vn "):
            norms.append(line)
        elif line.startswith("g "):
            current = line[2:].strip()
            groups[current] = []
        elif line.startswith("f ") and current:
            groups[current].append(line)

def parse_idx(token):
    # token like "12/5/3" or "12//3" or "12"
    parts = token.split("/")
    vi = int(parts[0]) if parts[0] else None
    ti = int(parts[1]) if len(parts) > 1 and parts[1] else None
    ni = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return vi, ti, ni

for name, faces in groups.items():
    v_map, t_map, n_map = {}, {}, {}
    out_faces = []
    for fline in faces:
        tokens = fline.split()[1:]
        new_tokens = []
        for tok in tokens:
            vi, ti, ni = parse_idx(tok)
            if vi not in v_map:
                v_map[vi] = len(v_map) + 1
            new_tok = str(v_map[vi])
            if ti is not None:
                if ti not in t_map:
                    t_map[ti] = len(t_map) + 1
                new_tok += f"/{t_map[ti]}"
                if ni is not None:
                    if ni not in n_map:
                        n_map[ni] = len(n_map) + 1
                    new_tok += f"/{n_map[ni]}"
            elif ni is not None:
                if ni not in n_map:
                    n_map[ni] = len(n_map) + 1
                new_tok += f"//{n_map[ni]}"
            new_tokens.append(new_tok)
        out_faces.append("f " + " ".join(new_tokens))

    outpath = os.path.join(OUTDIR, f"{name}.obj")
    with open(outpath, "w") as out:
        out.write(f"mtllib ../moonrock.mtl\n")
        for old_i in sorted(v_map, key=v_map.get):
            out.write(verts[old_i - 1] + "\n")
        for old_i in sorted(t_map, key=t_map.get):
            out.write(uvs[old_i - 1] + "\n")
        for old_i in sorted(n_map, key=n_map.get):
            out.write(norms[old_i - 1] + "\n")
        out.write("usemtl MoonRocks\n")
        for fl in out_faces:
            out.write(fl + "\n")
    print(f"{name}: {len(v_map)} verts, {len(faces)} faces -> {outpath}")
