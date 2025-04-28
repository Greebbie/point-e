"""
make_abo_bundle.py
------------------
把本地解压后的 ABO 数据拼成 dataset.jsonl
1) 只读取 CSV 中 B 列 == 'general' 的行
2) 若同时存在 point cloud 和 image zip，则写入一行 JSON
"""

import csv, json, pathlib
from tqdm import tqdm

CSV_CAP = pathlib.Path(r"Cap3D_automated_ABO.csv")
PC_DIR  = pathlib.Path(r"compressed_pcs_00\ABO_pcs")
IMG_DIR = pathlib.Path(r"compressed_imgs_perobj_00\Cap3D_ABO_renderimgs")

OUT_DIR   = pathlib.Path("abo_bundle"); OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_JSONL = OUT_DIR / "dataset.jsonl"

# ===== csv =====
rows = []
with CSV_CAP.open(newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for uid, typ, caption, *rest in reader:     # A, B, C,...
        if typ.strip().lower() != "general":
            continue
        rows.append((uid.strip(), caption.strip()))

print(f"CSV at general：{len(rows)}")

bundles = []
for uid, caption in tqdm(rows, desc="scanning local files"):
    ply_path = PC_DIR / f"{uid}.ply"
    zip_path = IMG_DIR / f"{uid}.zip"

    if ply_path.exists() and zip_path.exists():
        bundles.append({
            "uid": uid,
            "ply": str(ply_path.resolve()),
            "img_zip": str(zip_path.resolve()),
            "caption": caption
        })

print(f"ok {len(bundles)} 3 samples")

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for b in bundles:
        json.dump(b, f, ensure_ascii=False)
        f.write("\n")

print("yes：", OUT_JSONL.resolve())
