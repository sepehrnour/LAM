"""Quick analysis of the ARKit-to-FLAME mapping matrix."""
import numpy as np
import json

with open("assets/default_arkit_to_flame.json") as f:
    data = json.load(f)

M = np.array(data["matrix"], dtype=np.float32)
names = data["arkit_blendshape_names"]

used = np.where(np.any(M != 0, axis=0))[0]
print(f"Matrix shape: {M.shape}")
print(f"Used FLAME expression dims: {used.tolist()}")
print(f"Count: {len(used)} of {M.shape[1]}")
print(f"Max FLAME dim index: {int(used.max())}")
print(f"Nonzero entries: {np.count_nonzero(M)} of {M.size}")

zero_arkit = np.where(np.all(M == 0, axis=1))[0]
print(f"\nUnmapped ARKit shapes ({len(zero_arkit)}):")
for i in zero_arkit:
    print(f"  [{i}] {names[i]}")

print(f"\nMapped ARKit shapes ({52 - len(zero_arkit)}):")
for i in range(52):
    if i not in zero_arkit:
        nonzero_dims = np.where(M[i] != 0)[0]
        weights = [(int(d), float(M[i, d])) for d in nonzero_dims]
        print(f"  [{i}] {names[i]:25s} -> dims {weights}")
