#!/usr/bin/env python3
# extract_pointclouds.py
"""
Batch-convert DRACO/NOCS rendered frames into world-space point-clouds.

 • hard-coded intrinsics  (Blender 640×480 render: fx=888.89, fy=1000)
 • reads EXR depth via  OpenEXR-Py   OR   imageio-openexr
 • outputs   ground_truth_pointcloud.npy   (+ optional .ply)

Author: ChatGPT (o3)
-----------------------------------------------------------------------
"""

# ── USER SETTINGS ─────────────────────────────────────────────────────
CLASS_ROOT = (r"C:\Users\lvbab\OneDrive\Documents\GitHub"
              r"\DRACO-Weakly-Supervised-Dense-Reconstruction-And-Canonicalization-of-Objects"
              r"\outputs\02958343")                # change me
WRITE_PLY   = True                                 # False ➜ only .npy
# ---------------------------------------------------------------------

import os, sys, json, glob, argparse
import numpy as np
from PIL import Image

# ── fixed camera intrinsics coming from NOCS_render.py ───────────────
K = np.array([[888.8889, 0.     , 320.0],
              [0.     , 1000.   , 240.0],
              [0.     ,   0.   ,   1.0]], dtype=np.float32)
Kinv = np.linalg.inv(K)

# ───────────────────────── helpers ───────────────────────────────────
def load_depth_exr(path:str) -> np.ndarray:
    """Return H×W float32 depth image; tries OpenEXR-Py then imageio."""
    # ① OpenEXR-Py
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(path)
        dw  = exr.header()['dataWindow']
        W   = dw.max.x - dw.min.x + 1
        H   = dw.max.y - dw.min.y + 1
        chan = next(iter(exr.header()['channels']))  # e.g. 'R'
        pt   = Imath.PixelType(Imath.PixelType.FLOAT)
        depth = np.frombuffer(exr.channel(chan, pt), dtype=np.float32)
        return depth.reshape(H, W)
    except Exception:
        pass
    # ② imageio-openexr
    try:
        import imageio.v2 as iio
        d = iio.imread(path)
        if d.ndim == 3:
            d = d[...,0]
        return d.astype(np.float32)
    except Exception as e:
        sys.exit(f"❌ cannot read EXR {path}\n{e}")

def quat_to_mat(q):
    """Convert quaternion dict → 3×3 rotation matrix."""
    qw,qx,qy,qz = q["w"],q["x"],q["y"],q["z"]
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),     2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=np.float32)

def write_ply(path, pts:np.ndarray):
    with open(path,"w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        np.savetxt(f, pts, fmt="%.6f")

# ───────────────────────── main loop ─────────────────────────────────
def process_model_folder(folder:str):
    dpth = os.path.join(folder,"frame_00000000_Depth_00.exr")
    msk  = os.path.join(folder,"frame_00000000_Mask_00.png")
    pose = os.path.join(folder,"frame_00000000_CameraPose.json")
    outn = os.path.join(folder,"ground_truth_pointcloud.npy")

    if not (os.path.exists(dpth) and os.path.exists(msk) and os.path.exists(pose)):
        print("   ⚠  missing files – skipped")
        return

    # already done?
    if os.path.exists(outn):
        print("   ✅ already exists")
        return

    depth = load_depth_exr(dpth)
    mask  = np.array(Image.open(msk)) > 127

    H,W   = depth.shape
    i,j   = np.indices((H,W))
    pix   = np.stack([j,i,np.ones_like(j)],-1).reshape(-1,3)
    z     = depth.reshape(-1,1)
    pts_c = (pix @ Kinv.T) * z
    keep  = mask.reshape(-1) & np.isfinite(z[:,0]) & (z[:,0]>0)
    pts_c = pts_c[keep]

    P     = json.load(open(pose))
    R     = quat_to_mat(P["rotation"])
    t     = np.array([P["position"][k] for k in ("x","y","z")], dtype=np.float32)
    pts_w = (R @ pts_c.T).T + t

    np.save(outn, pts_w)
    print(f"   💾 saved {len(pts_w):,} pts → point_cloud.npy")

    if WRITE_PLY:
        write_ply(os.path.join(folder,"ground_truth_pointcloud.ply"), pts_w)
        print("   💾 wrote point_cloud.ply")

def main():
    root = os.path.abspath(CLASS_ROOT)
    if not os.path.isdir(root):
        sys.exit(f"Class root does not exist:\n{root}")

    subdirs = sorted(d for d in glob.glob(os.path.join(root,"*")) if os.path.isdir(d))
    print(f"🗂️  {len(subdirs)} model folders under {root}\n")

    for mdl in subdirs:
        print("🔄", os.path.basename(mdl))
        process_model_folder(mdl)

if __name__ == "__main__":
    main()
