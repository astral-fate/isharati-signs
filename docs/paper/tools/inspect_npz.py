"""Inspect the npz structure."""
import numpy as np
from pathlib import Path

kp_path = Path(r"D:\islam\gathring data\kfc_videos\R1 - الإصدار الأول\R1 - الإصدار الأول.keypoints.npz")
data = np.load(str(kp_path), allow_pickle=True)
for k in data.keys():
    v = data[k]
    print(f"  {k}: shape={v.shape if hasattr(v,'shape') else 'N/A'}, dtype={v.dtype}")
    if v.shape == ():
        print(f"    scalar value: {v.item()}")
    elif len(v.shape) > 0:
        print(f"    first few: {v[:3] if len(v) > 3 else v}")
