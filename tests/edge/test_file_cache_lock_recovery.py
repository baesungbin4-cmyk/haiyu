"""Lock recovery tests for OfflineEventCache (review S1/S5)."""

import os
import tempfile
from pathlib import Path

from edge.cache.file_cache import OfflineEventCache, _pid_is_alive


def test_pid_is_alive_returns_true_for_current_process() -> None:
    """The current process must be alive."""
    assert _pid_is_alive(os.getpid()) is True


def test_pid_is_alive_returns_false_for_nonexistent_pid() -> None:
    """A very high PID that cannot exist should be dead."""
    # Use a very large PID that's guaranteed not to exist
    dead_pid = 999999
    assert (
        _pid_is_alive(dead_pid) is False
    ), f"PID {dead_pid} should not exist on this system"


def test_stale_lock_is_broken_when_owner_pid_is_dead() -> None:
    """If the lock file's PID is dead, acquisition must succeed."""
    cache_dir = Path(tempfile.mkdtemp(prefix="haiyu_test_cache_"))
    try:
        cache = OfflineEventCache(
            cache_dir=cache_dir,
            lock_timeout_seconds=0.5,
        )
        # Create a fake lock file with a dead PID
        dead_pid = 999999
        cache.lock_path.write_text(str(dead_pid), encoding="ascii")

        # Acquisition should break the stale lock and succeed
        lock_fd = cache._acquire_process_lock()
        assert lock_fd >= 0
        cache._release_process_lock(lock_fd)

        # After release, the lock file should be gone
        assert not cache.lock_path.exists()
    finally:
        _cleanup_dir(cache_dir)


def test_live_process_lock_is_not_broken() -> None:
    """If the lock file's PID is alive, acquisition must time out."""
    cache_dir = Path(tempfile.mkdtemp(prefix="haiyu_test_cache_"))
    try:
        cache = OfflineEventCache(
            cache_dir=cache_dir,
            lock_timeout_seconds=0.2,
        )
        # Create a lock file with our own PID (alive!)
        cache.lock_path.write_text(str(os.getpid()), encoding="ascii")

        try:
            cache._acquire_process_lock()
        except TimeoutError as exc:
            msg = str(exc)
            assert "timed out" in msg
            assert str(cache.lock_path) in msg
            assert (
                str(os.getpid()) in msg
            ), "TimeoutError must include the holding PID (review S5)"
        else:
            raise AssertionError("Should have timed out — lock held by live process")

        # Lock file should still exist (we never broke it)
        assert cache.lock_path.exists()
    finally:
        cache.lock_path.unlink(missing_ok=True)
        _cleanup_dir(cache_dir)


def test_lock_contains_current_pid() -> None:
    """After acquisition, lock file must contain the acquiring PID."""
    cache_dir = Path(tempfile.mkdtemp(prefix="haiyu_test_cache_"))
    try:
        cache = OfflineEventCache(
            cache_dir=cache_dir,
            lock_timeout_seconds=0.5,
        )
        lock_fd = cache._acquire_process_lock()
        try:
            content = cache.lock_path.read_text(encoding="ascii").strip()
            assert content == str(os.getpid())
        finally:
            cache._release_process_lock(lock_fd)
    finally:
        _cleanup_dir(cache_dir)


def test_append_and_read_basic_workflow() -> None:
    """Smoke test: append event, read, remove — all within lock."""
    from edge.common.schemas import MqttEventEnvelope

    cache_dir = Path(tempfile.mkdtemp(prefix="haiyu_test_cache_"))
    try:
        cache = OfflineEventCache(cache_dir=cache_dir)
        event = MqttEventEnvelope(
            device_id="edge-01",
            timestamp=1_718_000_000.0,
            seq=1,
            event_type="alarm",
            topic="haiyu/edge-01/alarm",
            payload={"risk": "high"},
        )
        cache.append(event)
        all_events = cache.read_all()
        assert len(all_events) == 1
        assert all_events[0].seq == 1
        assert all_events[0].device_id == "edge-01"

        cache.remove_sent([event])
        assert len(cache.read_all()) == 0
    finally:
        _cleanup_dir(cache_dir)


def _cleanup_dir(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
