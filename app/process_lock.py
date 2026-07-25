from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO, Optional


class ProcessLock:
    """Prevents two web processes from consuming the same SQLite queue."""

    def __init__(self, path: Path):
        self.path = path
        self._handle: Optional[IO[str]] = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        os.chmod(self.path, 0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "检测到另一个叙枢进程正在使用同一数据目录；"
                "SQLite 队列模式只能启动一个应用进程"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
