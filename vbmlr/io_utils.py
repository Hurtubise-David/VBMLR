# vbmlr/io_utils.py
import os
import re
import numpy as np

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def write_csv_line(path: str, arr):
    s = ",".join([f"{float(x):.10g}" for x in arr])
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def read_features_file_cpp_robust(path: str, expected_dim: int = 8):
    """
    Red features ',;'.
    Extract floats per regex and reshape in NxD.
    """
    if not os.path.isfile(path):
        return None
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    if not txt.strip():
        return None

    nums = _FLOAT_RE.findall(txt)
    if len(nums) < expected_dim:
        return None

    vals = np.array([float(x) for x in nums], dtype=np.float64)
    n = vals.size // expected_dim
    if n <= 0:
        return None
    vals = vals[: n * expected_dim]
    return vals.reshape(n, expected_dim)
