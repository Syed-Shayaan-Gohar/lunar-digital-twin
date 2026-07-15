#!/usr/bin/env python3
import glob

for path in sorted(glob.glob("rocks_split/*.obj")):
    xs, ys, zs = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                xs.append(float(x)); ys.append(float(y)); zs.append(float(z))
    if not xs:
        print(f"{path}: NO VERTS")
        continue
    print(f"{path}:")
    print(f"  X: {min(xs):.3f} to {max(xs):.3f}  (extent {max(xs)-min(xs):.3f})")
    print(f"  Y: {min(ys):.3f} to {max(ys):.3f}  (extent {max(ys)-min(ys):.3f})")
    print(f"  Z: {min(zs):.3f} to {max(zs):.3f}  (extent {max(zs)-min(zs):.3f})")
