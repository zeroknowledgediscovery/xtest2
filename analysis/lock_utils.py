"""SHA-256 hashing helpers for the pre-outer protocol lock."""
import hashlib
import glob
import os


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def hash_many(paths):
    return {os.path.relpath(p): sha256_file(p) for p in paths if os.path.isfile(p)}


def list_source_files(analysis_dir):
    return sorted(glob.glob(os.path.join(analysis_dir, "*.py")))
