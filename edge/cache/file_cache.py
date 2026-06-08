"""File-backed offline cache for edge MQTT events."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge.common.schemas import MqttEventEnvelope


@dataclass(frozen=True)
class CachedEvent:
    """Cached MQTT event with a stable deduplication key."""

    event: MqttEventEnvelope

    @property
    def key(self) -> tuple[str, int]:
        return (self.event.device_id, self.event.seq)


class OfflineEventCache:
    """Append/read cache using prediction_logs.json and prediction_logs.tmp.

    Authentication status: this cache stores serialized payloads only.
    Encrypting cached data at rest is not implemented in this task.
    """

    def __init__(
        self,
        cache_dir: str | Path = "edge/cache",
        persistent_name: str = "prediction_logs.json",
        temp_name: str = "prediction_logs.tmp",
        lock_name: str = "prediction_logs.lock",
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.persistent_path = self.cache_dir / persistent_name
        self.temp_path = self.cache_dir / temp_name
        self.lock_path = self.cache_dir / lock_name
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._thread_lock = threading.RLock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        event: MqttEventEnvelope,
        temporary: bool = False,
    ) -> None:
        with self._locked():
            path = self.temp_path if temporary else self.persistent_path
            events = self._read_path(path)
            key = (event.device_id, event.seq)
            if key in {(item.device_id, item.seq) for item in events}:
                return
            events.append(event)
            self._write_path(path, events)

    def read_all(self) -> list[MqttEventEnvelope]:
        with self._locked():
            seen: set[tuple[str, int]] = set()
            merged: list[MqttEventEnvelope] = []
            for event in [
                *self._read_path(self.persistent_path),
                *self._read_path(self.temp_path),
            ]:
                key = (event.device_id, event.seq)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(event)
            merged.sort(key=lambda item: (item.timestamp, item.seq))
            return merged

    def remove_sent(self, sent_events: list[MqttEventEnvelope]) -> None:
        with self._locked():
            sent_keys = {(event.device_id, event.seq) for event in sent_events}
            for path in (self.persistent_path, self.temp_path):
                remaining = [
                    event
                    for event in self._read_path(path)
                    if (event.device_id, event.seq) not in sent_keys
                ]
                self._write_path(path, remaining)

    def clear(self) -> None:
        with self._locked():
            self._write_path(self.persistent_path, [])
            self._write_path(self.temp_path, [])

    @contextmanager
    def _locked(self) -> Any:
        with self._thread_lock:
            lock_fd = self._acquire_process_lock()
            try:
                yield
            finally:
                self._release_process_lock(lock_fd)

    def _acquire_process_lock(self) -> int:
        """Acquire an inter-process lock with stale-lock recovery.

        If a lock file already exists, reads the PID stored inside and
        checks whether the owning process is still alive via
        ``os.kill(pid, 0)``.  Dead-owner locks are unlinked and retried
        so that a crashed process never permanently blocks the cache
        (review H1 / M3).
        """
        deadline = time.time() + self.lock_timeout_seconds
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        while True:
            try:
                lock_fd = os.open(self.lock_path, flags)
                os.write(lock_fd, str(os.getpid()).encode("ascii"))
                return lock_fd
            except FileExistsError:
                if self._stale_lock_owner_dead():
                    self._break_stale_lock()
                    continue  # retry acquisition immediately
                if time.time() >= deadline:
                    owner_pid = self._read_lock_pid()
                    raise TimeoutError(
                        "offline cache lock acquisition timed out "
                        f"(lock: {self.lock_path}, "
                        f"owner PID: {owner_pid or 'unknown'}, "
                        f"timeout: {self.lock_timeout_seconds}s). "
                        "If the owning process is no longer running, "
                        f"manually delete {self.lock_path}."
                    )
                time.sleep(0.01)

    def _stale_lock_owner_dead(self) -> bool:
        """Return True if the lock file's PID is not a running process.

        Only breaks locks where we can read a PID **and** confirm it is
        dead.  Unreadable PIDs (corrupt lock file, non-numeric content)
        are assumed valid to avoid breaking locks held by a different
        cache implementation.
        """
        pid = self._read_lock_pid()
        if pid is None:
            return False  # unreadable → assume valid
        return not _pid_is_alive(pid)

    def _read_lock_pid(self) -> int | None:
        """Read the PID stored in the lock file, or None."""
        try:
            content = self.lock_path.read_text(encoding="ascii").strip()
            return int(content)
        except (ValueError, OSError):
            return None

    def _break_stale_lock(self) -> None:
        """Remove a lock file whose owning process is no longer alive."""
        owner_pid = self._read_lock_pid()
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            return
        if owner_pid is not None:
            print(
                f"Stale lock {self.lock_path} from dead PID {owner_pid} "
                f"removed — cache recovered."
            )

    def _release_process_lock(self, lock_fd: int) -> None:
        os.close(lock_fd)
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            return

    def _read_path(self, path: Path) -> list[MqttEventEnvelope]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            backup_path = path.with_suffix(path.suffix + ".corrupted")
            path.replace(backup_path)
            return []
        if not isinstance(raw, list):
            raise ValueError(f"{path.name} must contain a JSON list")
        return [self._event_from_dict(item) for item in raw]

    def _write_path(self, path: Path, events: list[MqttEventEnvelope]) -> None:
        temp_path = path.with_suffix(path.suffix + ".atomic")
        temp_path.write_text(
            json.dumps(
                [event.to_dict() for event in events],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _event_from_dict(value: dict[str, Any]) -> MqttEventEnvelope:
        return MqttEventEnvelope(
            device_id=value["device_id"],
            timestamp=value["timestamp"],
            seq=value["seq"],
            event_type=value["event_type"],
            topic=value["topic"],
            payload=value["payload"],
            qos=value.get("qos", 1),
            retain=value.get("retain", False),
        )


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with the given *pid* exists.

    Uses ``os.kill(pid, 0)`` and treats access-denied checks as alive. Returns
    False only when the PID is confirmed absent.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        winerror = getattr(exc, "winerror", None)
        if os.name == "nt" and winerror == 87:
            return False
        if os.name == "nt" and winerror == 5:
            return True
        return True
    return True


__all__ = ["CachedEvent", "OfflineEventCache"]
