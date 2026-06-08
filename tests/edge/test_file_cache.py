from edge.cache.file_cache import OfflineEventCache
from edge.common.schemas import MqttEventEnvelope


def _event(seq: int, timestamp: float | None = None) -> MqttEventEnvelope:
    return MqttEventEnvelope(
        device_id="edge-01",
        timestamp=float(seq) if timestamp is None else timestamp,
        seq=seq,
        event_type="alarm",
        topic="haiyu/edge-01/alarm",
        payload={"seq": seq},
        qos=1,
        retain=False,
    )


def test_cache_write_read(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)
    event = _event(1)

    cache.append(event)

    assert cache.read_all()[0].to_dict() == event.to_dict()


def test_cache_survives_restart_simulation(tmp_path) -> None:
    first = OfflineEventCache(tmp_path)
    first.append(_event(1))

    restarted = OfflineEventCache(tmp_path)

    assert [event.seq for event in restarted.read_all()] == [1]


def test_resend_order_preserved_across_persistent_and_tmp(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)
    cache.append(_event(2, timestamp=20.0), temporary=True)
    cache.append(_event(1, timestamp=10.0))
    cache.append(_event(3, timestamp=30.0))

    assert [event.seq for event in cache.read_all()] == [1, 2, 3]


def test_cache_avoids_duplicate_sequence_ids(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)

    cache.append(_event(1))
    cache.append(_event(1))

    assert [event.seq for event in cache.read_all()] == [1]


def test_corrupted_cache_file_is_backed_up_and_reset(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)
    cache.persistent_path.write_text("{bad json", encoding="utf-8")

    assert cache.read_all() == []
    assert not cache.persistent_path.exists()
    assert (tmp_path / "prediction_logs.json.corrupted").exists()


def test_cache_append_ignores_duplicate_across_files_on_read(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)
    cache.append(_event(1))
    cache.append(_event(1), temporary=True)

    assert [event.seq for event in cache.read_all()] == [1]


def test_remove_sent_partial_removal(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)
    first = _event(1)
    second = _event(2)
    third = _event(3)
    cache.append(first)
    cache.append(second)
    cache.append(third, temporary=True)

    cache.remove_sent([second])

    assert [event.seq for event in cache.read_all()] == [1, 3]


def test_temporary_events_are_isolated_to_tmp_file(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)

    cache.append(_event(4), temporary=True)

    assert cache.persistent_path.exists() is False
    assert cache.temp_path.exists() is True
    assert [event.seq for event in cache.read_all()] == [4]


def test_cache_lock_file_is_released_after_operations(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)

    cache.append(_event(5))
    assert cache.lock_path.exists() is False
    cache.read_all()
    assert cache.lock_path.exists() is False


def test_cache_lock_timeout_when_another_process_holds_lock(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path, lock_timeout_seconds=0.01)
    cache.lock_path.write_text("held", encoding="ascii")

    try:
        cache.append(_event(6))
    except TimeoutError as exc:
        assert "lock" in str(exc)
    else:
        raise AssertionError("expected lock timeout")
