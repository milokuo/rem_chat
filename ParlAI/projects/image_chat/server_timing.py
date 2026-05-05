# -*- coding: utf-8 -*-
"""
server_timing.py

Timing helpers for rem_chat server.
Extracted from server_updated_zhengxuan.py (Phase 2 refactor).
"""

import datetime
import json
import os
import time
from typing import Any, Callable, Optional, Tuple

_TIMING_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_log.jsonl")


def _timed_worker(fn: Callable, *args: Any, **kwargs: Any) -> Tuple[Any, int]:
    """Wrap a callable so it returns (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round((time.perf_counter() - t0) * 1000)


def _log_timing(record: dict, log_path: Optional[str] = None) -> None:
    """Append a JSON record with a timestamp to the timing log."""
    record["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    path = log_path if log_path is not None else _TIMING_LOG
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[Timing] Log write failed: {e}")
