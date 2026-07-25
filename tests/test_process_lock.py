from pathlib import Path

import pytest

from app.process_lock import ProcessLock


def test_process_lock_rejects_second_instance(tmp_path: Path):
    first = ProcessLock(tmp_path / ".worker.lock")
    second = ProcessLock(tmp_path / ".worker.lock")
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="另一个 novelAI 进程"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()
