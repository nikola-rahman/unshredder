"""Load pickled .npy files (np.save of a dict) from untrusted sources without executing code.

Only the classes needed to rebuild plain NumPy arrays and containers are allowed; any
other global referenced by the pickle raises instead of being imported.
"""

from __future__ import annotations

import pickle

import numpy as np
from numpy.lib import format as npy_format

_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("collections", "OrderedDict"),
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("_codecs", "encode"),  # how protocol-2 pickles store byte strings; str -> bytes only
}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load(path: str):
    with open(path, "rb") as fh:
        version = npy_format.read_magic(fh)
        npy_format._check_version(version) if hasattr(npy_format, "_check_version") else None
        shape, _, dtype = (npy_format.read_array_header_1_0(fh) if version == (1, 0)
                           else npy_format.read_array_header_2_0(fh))
        if dtype != np.dtype("O"):
            fh.seek(0)
            return np.load(fh, allow_pickle=False)
        obj = _Restricted(fh, encoding="latin1").load()
    # np.save of a dict stores a 0-d object array wrapping it
    return obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj
