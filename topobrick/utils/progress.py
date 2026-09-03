from __future__ import annotations

import sys
import time

_START_TIME = time.time()


def log(message: str, *, flush: bool = True) -> None:
    """Write an elapsed-time progress message to stderr."""
    elapsed = time.time() - _START_TIME
    print(f"[{elapsed:8.1f}s] {message}", flush=flush, file=sys.stderr)
