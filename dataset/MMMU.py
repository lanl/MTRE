import os, glob, uuid, hashlib, re
import pandas as pd

class BaseDataset:
    def __init__(self): ...

# # --- helpers for options ---
# def _parse_options(text):
#     if text is None: return [], []
#     s = str(text).strip().replace('\r\n', '\n')
#     matches = list(re.finditer(r'(^|\n)\s*([A-G])\s*[\.\)]\s*', s))
#     if not matches:
#         parts = [p.strip() for p in s.split('\n') if p.strip()]
#         labels = [chr(ord('A') + i) for i in range(len(parts))]
#         return labels, parts
#     labels, choices = [], []
#     for i, m in enumerate(matches):
#         label, start = m.group(2), m.end()
#         end = matches[i+1].start() if i+1 < len(matches) else len(s)
#         choices.append(s[start:end].strip()); labels.append(label)
#     return labels, choices
import re, json, ast

def _parse_options(text):
    if text is None:
        return [], []

    # If it's already a list/tuple, just label it.
    if isinstance(text, (list, tuple)):
        choices = [str(x).strip() for x in text]
        labels = [chr(ord('A') + i) for i in range(len(choices))]
        return labels, choices

    s = str(text).strip().replace('\r\n', '\n')

    # If it's a JSON / Python-list string, parse it.
    if s.startswith('[') and s.endswith(']'):
        try:
            arr = json.loads(s)
        except json.JSONDecodeError:
            arr = ast.literal_eval(s)  # safer than eval for Python literals
        choices = [str(x).strip() for x in arr]
        labels = [chr(ord('A') + i) for i in range(len(choices))]
        return labels, choices

    # Look for A) / A. style labels; allow beyond G just in case
    matches = list(re.finditer(r'(^|\n)\s*([A-Z])\s*[\.\)]\s*', s))

    if not matches:
        parts = [p.strip() for p in s.split('\n') if p.strip()]
        labels = [chr(ord('A') + i) for i in range(len(parts))]
        return labels, parts

    labels, choices = [], []
    for i, m in enumerate(matches):
        label, start = m.group(2), m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(s)
        choices.append(s[start:end].strip())
        labels.append(label)
    return labels, choices


# def _answer_to_index(ans_str, labels, choices):
#     print(ans_str)
#     print(':::::')
#     print(labels)
#     print(':::')
#     print(choices)
#     if ans_str is None: return None
#     a = str(ans_str).strip()
#     if len(a)==1 and a.upper() in labels: return labels.index(a.upper())
#     for i,ch in enumerate(choices):
#         if a.lower()==ch.lower(): return i
#     return None

def _answer_to_index(ans_str, labels, choices):
    if ans_str is None:
        return None
    a = str(ans_str).strip()
    try:
        return labels.index(a)
    except ValueError:
        return None


# --- main ---
class MMMUParquetDataset(BaseDataset):
    """
    MMMU subject loader (e.g., .../MMMU/Accounting). Handles image_1..image_7 dicts with bytes/url/path.
    """
    def __init__(self, prompter, subject_dir, image_mode="first",
                 image_root=None, cache_dir=None, response_type="mc",
                 alt_image_roots=None, split="test"):
        super().__init__()
        self.prompter = prompter
        self.subject_dir = subject_dir
        self.image_mode = image_mode
        self.image_root = image_root or subject_dir
        self.cache_dir = cache_dir or os.path.join(subject_dir, "_img_cache")
        self.alt_image_roots = alt_image_roots or []
        os.makedirs(self.cache_dir, exist_ok=True)

        parq = sorted(glob.glob(os.path.join(subject_dir, f"{split}-*.parquet")))
        if not parq:
            raise FileNotFoundError(f"No parquet files in {subject_dir} for split={split}")
        cols = ["id","question","options","explanation",
                "image_1","image_2","image_3","image_4","image_5","image_6","image_7",
                "img_type","answer","topic_difficulty","question_type","subfield"]
        dfs = [pd.read_parquet(p, columns=[c for c in cols if c]) for p in parq]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(self.df)} rows from {len(parq)} shard(s) in {subject_dir}")

        # quick peek
        a = self.df.iloc[0].to_dict()
        a["image_1"] = {k:type(v).__name__ for k,v in a["image_1"].items()} if isinstance(a.get("image_1"),dict) else type(a.get("image_1")).__name__
        print("🔎 first row image_1 keys/types:", a["image_1"])

    # --- bytes→file, url/path handling ---
    @staticmethod
    def _guess_ext_from_magic(b: bytes) -> str:
        if b.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
        if b.startswith((b"\xff\xd8\xff\xdb", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xe1", b"\xff\xd8\xff\xe2")): return ".jpg"
        if b.startswith(b"RIFF") and b[8:12]==b"WEBP": return ".webp"
        if b[:6] in (b"GIF87a", b"GIF89a"): return ".gif"
        if b.startswith(b"BM"): return ".bmp"
        return ".img"

    def _bytes_to_cache(self, b: bytes, prefer_name=None) -> str:
        # Stable content hash filename avoids duplicates
        h = hashlib.sha1(b).hexdigest()[:16]
        ext = (os.path.splitext(prefer_name)[1] if prefer_name else None) or self._guess_ext_from_magic(b)
        path = os.path.join(self.cache_dir, f"{h}{ext}")
        if not os.path.exists(path):
            with open(path, "wb") as f: f.write(b)
        return path

    def _extract_from_dict(self, d):
        if d is None: return None
        # 1) Prefer embedded bytes
        for kb in ("bytes","data_bytes","content"):
            v = d.get(kb)
            if isinstance(v, (bytes, bytearray)) and len(v)>0:
                prefer = d.get("filename") or d.get("file_name") or d.get("name")
                return self._bytes_to_cache(v, prefer_name=prefer)
        # 2) Then any useful string (url or path)
        for ks in ("path","filepath","file_path","url","image_url","filename","file_name","name"):
            v = d.get(ks)
            if isinstance(v, str) and v:
                return v
        # 3) Nested dicts
        for kn in ("image","data","value"):
            v = d.get(kn)
            if isinstance(v, dict):
                out = self._extract_from_dict(v)
                if out: return out
        return None

    def _resolve_one_image(self, val):
        # URL
        if isinstance(val, str) and val.startswith(("http://","https://")):
            return val
        # Absolute local path
        if isinstance(val, str) and os.path.isabs(val):
            return val if os.path.exists(val) else val  # keep for debugging even if missing
        # Relative filename/path
        if isinstance(val, str):
            cand = os.path.join(self.image_root, val)
            if os.path.exists(cand): return cand
            for root in self.alt_image_roots:
                p = os.path.join(root, val)
                if os.path.exists(p): return p
            return cand  # return best-guess even if not present (helps you see what's missing)
        # Dict
        if isinstance(val, dict):
            extracted = self._extract_from_dict(val)
            if extracted is None: return None
            return self._resolve_one_image(extracted)
        return None

    def _collect_images(self, row):
        imgs = []
        for i in range(1,8):
            key = f"image_{i}"
            if key in row:
                path = self._resolve_one_image(row[key])
                if path: imgs.append(path)
        return imgs

    def get_data(self):
        records = []
        for _, r in self.df.iterrows():
            labels, choices = _parse_options(r.get("options",""))
            answer_idx = _answer_to_index(r.get("answer"), labels, choices)
            img_paths = self._collect_images(r)

            if self.image_mode == "list":
                rec = {
                    "id": r["id"], "img_paths": img_paths,
                    "img_path": img_paths[0] if img_paths else None,
                    "question": r["question"],
                    "options": choices, "option_labels": labels,
                    "answer_index": answer_idx,
                    "label": choices[answer_idx] if (answer_idx is not None and answer_idx<len(choices)) else r.get("answer"),
                    "original_question": r["question"],
                    "meta": {
                        "img_type": r.get("img_type"), "question_type": r.get("question_type"),
                        "subfield": r.get("subfield"), "topic_difficulty": r.get("topic_difficulty"),
                        "explanation": r.get("explanation"),
                    }
                }
                records.append(rec)

            elif self.image_mode == "expand":
                if not img_paths: img_paths = [None]
                for j, p in enumerate(img_paths, start=1):
                    rec = {
                        "id": f"{r['id']}#img{j}", "img_path": p,
                        "question": r["question"],
                        "options": choices, "option_labels": labels,
                        "answer_index": answer_idx,
                        "label": choices[answer_idx] if (answer_idx is not None and answer_idx<len(choices)) else r.get("answer"),
                        "original_question": r["question"],
                        "meta": {
                            "img_idx": j, "img_type": r.get("img_type"),
                            "question_type": r.get("question_type"), "subfield": r.get("subfield"),
                            "topic_difficulty": r.get("topic_difficulty"), "explanation": r.get("explanation"),
                        }
                    }
                    records.append(rec)

            else:  # "first"
                question_text = r["question"] + (f'\nOptions: {r["options"]}' if r.get("options") else "")
                p = img_paths[0] if img_paths else None
                rec = {
                    "id": r["id"], "img_path": p,
                    "question": question_text,
                    "options": choices, "option_labels": labels,
                    "answer_index": answer_idx,
                    "label": choices[answer_idx] if (answer_idx is not None and answer_idx<len(choices)) else r.get("answer"),
                    "original_question": r["question"],
                    "question_type": r['question_type'],
                    "answer": r['answer'],
                    "meta": {
                        "img_type": r.get("img_type"), "question_type": r.get("question_type"),
                        "subfield": r.get("subfield"), "topic_difficulty": r.get("topic_difficulty"),
                        "explanation": r.get("explanation"),
                    }
                }
                records.append(rec)

        return records, ["id"]


# dataset = MMMUParquetDataset(
#     prompter=None,
#     subject_dir="/lustre/scratch5/gzollicoffer/rebuttal/MMMU/Accounting",
#     split="test",
#     image_mode="first",
#     cache_dir="/lustre/scratch5/gzollicoffer/rebuttal/MMMU/Accounting/_img_cache"  # ensure this matches what you check
# )
# data, _ = dataset.get_data()

# import glob
# files = glob.glob(os.path.join(dataset.cache_dir, "*"))
# print("Cache dir:", dataset.cache_dir, "num files:", len(files))
# print("Example path:", data[0]["img_path"])