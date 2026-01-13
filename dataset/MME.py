import os, glob, uuid, hashlib
import pandas as pd

# Minimal stub
class BaseDataset:
    def __init__(self): ...

class MMEParquetDataset(BaseDataset):
    """
    Loads MME parquet shards and returns records with a usable 'img_path'.
    Priority:
      1) If a sidecar CSV map (question_id -> img_path) exists AND file exists -> use it
      2) Else if parquet 'image' dict has raw bytes -> write to cache -> use that path
      3) Else if parquet 'image' dict or string has URL -> return URL (let loader download), OR
         if it's a filename -> join with image_root
    """
    def __init__(self,
                 prompter,
                 data_root="./data/MME/data",  # UPDATE: Set to your MME data path
                 map_dir=None,                   # where your *.paths.csv live (default: images dir or data_root)
                 image_root=None,                # if image is just a filename
                 cache_dir=None,                 # where to write bytes if present
                 split="test",
                 response_type="oe"):
        super().__init__()
        self.prompter = prompter
        self.data_root = data_root
        self.image_root = image_root or data_root
        self.cache_dir = cache_dir or os.path.join(self.data_root, "_img_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1) read parquet shards
        pq_glob = os.path.join(data_root, f"{split}-*.parquet")
        parquet_files = sorted(glob.glob(pq_glob))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {pq_glob}")

        cols = ["question_id", "image", "question", "answer", "category"]
        dfs = [pd.read_parquet(f, columns=cols) for f in parquet_files]
        self.df = pd.concat(dfs, ignore_index=True)

        # 2) load sidecar csv maps if present
        #    e.g., you showed files like: test-00000-of-00004-...parquet.paths.csv
        self.id2path = {}
        default_map_dir = map_dir or os.path.join(os.path.dirname(data_root), "images")
        csv_candidates = sorted(glob.glob(os.path.join(default_map_dir, "*.paths.csv"))) \
                         + sorted(glob.glob(os.path.join(data_root, "*.paths.csv")))
        for csvf in csv_candidates:
            try:
                m = pd.read_csv(csvf)
                if {"question_id","img_path"}.issubset(m.columns):
                    for qid, p in zip(m["question_id"], m["img_path"]):
                        if isinstance(qid, str) and isinstance(p, str):
                            self.id2path[qid] = p
            except Exception:
                pass  # ignore bad csvs

        print(f"✅ Loaded {len(self.df)} rows from {len(parquet_files)} shards")
        print(f"🔎 Loaded {len(self.id2path)} mapped img paths from CSV")

    # ---------- helpers ----------
    @staticmethod
    def _guess_ext_from_magic(b: bytes) -> str:
        if b.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
        if b.startswith((b"\xff\xd8\xff\xdb", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\xff\xd8\xff\xe2")): return ".jpg"
        if b.startswith(b"RIFF") and b[8:12] == b"WEBP": return ".webp"
        if b.startswith(b"BM"): return ".bmp"
        if b[:6] in (b"GIF87a", b"GIF89a"): return ".gif"
        return ".img"

    def _bytes_to_cache(self, b: bytes) -> str:
        h = hashlib.sha1(b).hexdigest()[:16]
        ext = self._guess_ext_from_magic(b)
        fp = os.path.join(self.cache_dir, f"{h}{ext}")
        if not os.path.exists(fp):
            with open(fp, "wb") as f:
                f.write(b)
        return fp

    def _extract_from_image_field(self, img_val):
        """
        Resolve a parquet 'image' field (dict or str) to a usable path/URL.
        """
        # direct string
        if isinstance(img_val, str):
            # absolute path -> use as-is
            if os.path.isabs(img_val):
                return img_val
            # URL
            if img_val.startswith(("http://","https://")):
                return img_val
            # relative filename
            return os.path.join(self.image_root, img_val)

        # dict with bytes / url / path
        if isinstance(img_val, dict):
            # 1) prefer embedded bytes
            b = img_val.get("bytes") or img_val.get("content") or img_val.get("data_bytes")
            if isinstance(b, (bytes, bytearray)) and len(b) > 0:
                return self._bytes_to_cache(b)

            # 2) then any useful string
            for k in ["path","filepath","file_path","url","image_url","filename","file_name","name"]:
                v = img_val.get(k)
                if isinstance(v, str) and v:
                    if os.path.isabs(v) or v.startswith(("http://","https://")):
                        return v
                    return os.path.join(self.image_root, v)
        return None

    # ---------- main ----------
    def get_data(self):
        missing_map, used_map, wrote_bytes, unresolved = 0,0,0,0
        data = []

        for _, row in self.df.iterrows():
            qid = row["question_id"]
            # 1) prefer sidecar map if it points to an existing file
            mapped = self.id2path.get(qid)
            if isinstance(mapped, str) and os.path.isabs(mapped) and os.path.exists(mapped):
                img_path = mapped
                used_map += 1
            else:
                if mapped:
                    missing_map += 1  # mapped but file not present
                # 2) fall back to parquet image field
                img_path = self._extract_from_image_field(row["image"])
                if img_path is None:
                    unresolved += 1

            data.append({
                "img_path": img_path,
                "question": row["question"],
                "label": row["answer"],
                "original_question": row["question"],
                "question_id": qid,
                "category": row.get("category"),
            })

        print(f"📊 img_path stats -> used_map:{used_map}, mapped_missing:{missing_map}, unresolved:{unresolved}")
        return data, ["question_id"]



# dataset = MMEParquetDataset(prompter=None)
# data, keys = dataset.get_data()

# print("Number of samples:", len(data))
# print("First example:", data[0])
