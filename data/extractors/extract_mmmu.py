import os, uuid, hashlib
import numpy as np

def _bytes_like(obj):
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
        return obj.tobytes()
    try:
        import pyarrow as pa
        if isinstance(obj, pa.Buffer):
            return obj.to_pybytes()
    except Exception:
        pass
    return None

def _guess_ext_from_magic(b: bytes) -> str:
    if b.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
    if b.startswith((b"\xff\xd8\xff",)):   return ".jpg"
    if b.startswith(b"RIFF") and b[8:12]==b"WEBP": return ".webp"
    if b[:6] in (b"GIF87a", b"GIF89a"): return ".gif"
    if b.startswith(b"BM"): return ".bmp"
    return ".img"

def _bytes_to_cache(cache_dir, data: bytes, prefer_name=None):
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1(data).hexdigest()[:16]
    ext = (os.path.splitext(prefer_name)[1] if prefer_name else None) or _guess_ext_from_magic(data)
    path = os.path.join(cache_dir, f"{h}{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f: f.write(data)
    return path

def _extract_from_dict(self, d):
    if d is None:
        return None
    # 1) Prefer bytes-like
    for k in ("bytes","data_bytes","content"):
        v = d.get(k)
        b = _bytes_like(v)
        if b:
            prefer = d.get("filename") or d.get("file_name") or d.get("name")
            return _bytes_to_cache(self.cache_dir, b, prefer_name=prefer)
    # 2) Then any useful string
    for k in ("path","filepath","file_path","url","image_url","filename","file_name","name"):
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    # 3) Nested dicts
    for k in ("image","data","value"):
        v = d.get(k)
        if isinstance(v, dict):
            out = self._extract_from_dict(v)
            if out:
                return out
    return None
