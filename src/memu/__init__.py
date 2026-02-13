from __future__ import annotations

from typing import Callable, Optional

_hello_from_bin: Optional[Callable[[], str]]

try:
    from memu._core import hello_from_bin as _hello_from_bin
except Exception:
    _hello_from_bin = None


def _rust_entry() -> str:
    if _hello_from_bin is None:
        msg = (
            "memu._core extension is not available. "
            "Install with Python >= 3.13 and build the Rust extension, "
            "or use PYTHONPATH for pure-Python usage."
        )
        raise RuntimeError(msg)
    return _hello_from_bin()
