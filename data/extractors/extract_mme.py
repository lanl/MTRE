#!/usr/bin/env python3
"""
MME Dataset Image Extractor

Extracts images from MME parquet files and saves them as individual image files.

Usage:
    python extract_mme.py --src ./data/MME/data --out ./data/MME/images

Arguments:
    --src: Directory containing MME parquet files (test-*.parquet)
    --out: Output directory for extracted images
"""
import os, glob, hashlib, argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Extract images from MME parquet files")
    parser.add_argument("--src", default="./data/MME/data",
                        help="Source directory containing parquet files")
    parser.add_argument("--out", default="./data/MME/images",
                        help="Output directory for extracted images")
    return parser.parse_args()

args = parse_args()
SRC_DIR = args.src
OUT_DIR = args.out
os.makedirs(OUT_DIR, exist_ok=True)

def guess_ext_from_magic(b: bytes) -> str:
    if b.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
    if b.startswith((b"\xff\xd8\xff\xdb", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\xff\xd8\xff\xe2")): return ".jpg"
    if b.startswith(b"RIFF") and b[8:12] == b"WEBP": return ".webp"
    if b.startswith(b"BM"): return ".bmp"
    if b[:6] in (b"GIF87a", b"GIF89a"): return ".gif"
    return ".img"  # fallback

def write_bytes_get_path(img_dict):
    """
    Returns a local filepath:
    - If dict has raw bytes -> write to OUT_DIR and return the path
    - If dict has a URL/path string -> return that string (you can later download/copy)
    """
    if not isinstance(img_dict, dict):
        return None

    # Prefer embedded bytes
    b = img_dict.get("bytes") or img_dict.get("content") or img_dict.get("data_bytes")
    if isinstance(b, (bytes, bytearray)) and len(b) > 0:
        # stable name: sha1 of bytes
        h = hashlib.sha1(b).hexdigest()[:16]
        ext = guess_ext_from_magic(b)
        fp = os.path.join(OUT_DIR, f"{h}{ext}")
        if not os.path.exists(fp):
            with open(fp, "wb") as f:
                f.write(b)
        return fp

    # Else fall back to string path/url if present (optional)
    for k in ["path", "filepath", "file_path", "url", "image_url", "filename", "file_name", "name"]:
        v = img_dict.get(k)
        if isinstance(v, str) and v:
            return v

    return None

# Walk all shards and materialize
parqs = sorted(glob.glob(os.path.join(SRC_DIR, "test-*.parquet")))
total, wrote, missing = 0, 0, 0
for p in parqs:
    df = pd.read_parquet(p, columns=["question_id", "image", "question", "answer", "category"])
    local_paths = []
    for _, row in df.iterrows():
        total += 1
        path = write_bytes_get_path(row["image"])
        if path and os.path.isabs(path) and os.path.exists(path):
            wrote += 1
        elif path and (path.startswith("http://") or path.startswith("https://") or not os.path.isabs(path)):
            # URL or relative path: not written locally here (optional to handle later)
            pass
        else:
            missing += 1
        local_paths.append(path)
    # Optionally save a sidecar csv mapping id -> local path
    out_map = os.path.join(OUT_DIR, os.path.basename(p) + ".paths.csv")
    pd.DataFrame({"question_id": df["question_id"], "img_path": local_paths}).to_csv(out_map, index=False)

print(f"done. rows={total}, wrote_local_files≈{wrote}, unresolved≈{missing}")