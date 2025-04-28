"""
integrate_abo_dataset.py
------------------------
把 make_abo_bundle.py 产出的 dataset.jsonl 进一步整合：
   pointclouds/uid.ply
   images/uid.png
   captions/uid.txt
并写出 final_dataset.jsonl
"""

import shutil, zipfile, io, pathlib, orjson, tqdm
import open3d as o3d, plyfile, random
import numpy as np

SRC_JSONL = pathlib.Path("abo_bundle/dataset.jsonl")   
ROOT_OUT  = pathlib.Path("abo_integrated")            
PC_DIR    = ROOT_OUT / "pointclouds"
IMG_DIR   = ROOT_OUT / "images"
TXT_DIR   = ROOT_OUT / "captions"

N_POINTS = 4096          # 与 Point-E base40M 一致, upsample之前就只调base了
SAVE_COLOR = True 


def ply2npz(ply_path, out_path, n_points=4096):
    ply = plyfile.PlyData.read(str(ply_path))
    v   = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)

    has_color = {"red", "green", "blue"} <= set(v.dtype.names)
    if has_color:
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)

    # ---------- 归一化 ----------
    xyz -= xyz.mean(0)
    xyz /= np.linalg.norm(xyz, axis=1).max()

    # ---------- 采样 ----------
    if len(xyz) > n_points:                  # 随机下采样
        idx = np.random.choice(len(xyz), n_points, replace=False)
        xyz = xyz[idx]
        if has_color:
            rgb = rgb[idx]
    elif len(xyz) < n_points:                # 用 Open3D Farthest 补采样
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if has_color:
            pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
        pcd = pcd.farthest_point_down_sample(n_points)
        xyz = np.asarray(pcd.points, np.float32)
        if has_color:
            rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    # ---------- 保存 ----------
    out = {"coords": xyz}
    if has_color:
        out["colors"] = rgb
    np.savez_compressed(out_path, **out)


for d in (PC_DIR, IMG_DIR, TXT_DIR):
    d.mkdir(parents=True, exist_ok=True)



# -------- 1. 逐行读取旧 jsonl，开始整合 ----------
final_records = []
with SRC_JSONL.open("r", encoding="utf-8") as f:
    for line in tqdm.tqdm(f, desc="integrating"):
        rec = orjson.loads(line)
        uid = rec["uid"]

        # 点云
        src_ply = pathlib.Path(rec["ply"])
        dst_ply = PC_DIR / f"{uid}.ply"
        if not dst_ply.exists():
            npz_path = PC_DIR / f"{uid}.npz"
            ply2npz(src_ply, npz_path)

        # 图片提00000
        src_zip = pathlib.Path(rec["img_zip"])
        dst_png = IMG_DIR / f"{uid}.png"
        if not dst_png.exists():
            with zipfile.ZipFile(src_zip) as z:
                img_bytes = z.read("00000.png")  # 想多视图就循环 z.namelist()
            dst_png.write_bytes(img_bytes)

        # Caption 
        dst_txt = TXT_DIR / f"{uid}.txt"
        if not dst_txt.exists():
            dst_txt.write_text(rec["caption"], encoding="utf-8")

        # metadata
        final_records.append({
            "uid": uid,
            "pointcloud": str(npz_path.relative_to(ROOT_OUT)),
            "image": str(dst_png.relative_to(ROOT_OUT)),
            "caption_file": str(dst_txt.relative_to(ROOT_OUT))
        })


OUT_JSONL = ROOT_OUT / "final_dataset.jsonl"
with OUT_JSONL.open("w", encoding="utf-8") as f:
    for r in final_records:
        f.write(orjson.dumps(r).decode() + "\n")

print(f"yes 共整合 {len(final_records)} 个样本 → {OUT_JSONL}")
