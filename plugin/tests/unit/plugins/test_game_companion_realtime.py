from pathlib import Path
import base64

from PIL import Image

from plugin.plugins.game_companion.core.frame_analyzer import analyze_frame_data_url
from plugin.plugins.game_companion.core.realtime import RealtimeInsightSession
from plugin.plugins.game_companion.core.replay import (
    append_snapshot,
    build_neko_context_packet,
    build_snapshot,
    build_training_prompt,
    clear_snapshots,
    dequeue_neko_context_packet,
    enqueue_neko_context_packet,
    list_neko_context_queue,
    load_snapshots,
)


class _Store:
    def __init__(self) -> None:
        self.values = {}

    def _read_value(self, key, default=None):
        return self.values.get(key, default)

    def _write_value(self, key, value):
        self.values[key] = value


def test_analyze_frame_data_url_reads_tft_png(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (1920, 1080), color=(4, 5, 6)).save(image_path)
    data_url = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = analyze_frame_data_url("tft", data_url)

    assert payload["success"] is True
    assert payload["profile"] == "tft"
    assert payload["source"]["type"] == "image_data_url"
    assert payload["source"]["path"] is None


def test_analyze_frame_data_url_reads_generic_png(tmp_path: Path) -> None:
    image_path = tmp_path / "generic.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(image_path)
    data_url = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = analyze_frame_data_url("generic", data_url)

    assert payload["success"] is True
    assert payload["profile"] == "generic"
    assert payload["source"]["type"] == "image_data_url"
    assert payload["source"]["path"] is None
    assert payload["vision"]["source"]["type"] == "image_data_url"
    assert payload["vision"]["source"]["path"] is None
    assert payload["vision"]["scene"]["label"] == "unknown"


def test_analyze_frame_data_url_preserves_origin_without_temp_path(tmp_path: Path) -> None:
    image_path = tmp_path / "generic_video_frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(image_path)
    data_url = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    source_context = {
        "type": "video_frame",
        "profile_id": "generic",
        "video_path": str((tmp_path / "match.mp4").resolve()),
        "ordinal": 4,
        "frame_index": 240,
        "timestamp_seconds": 24.0,
    }

    payload = analyze_frame_data_url("generic", data_url, source_context=source_context)

    assert payload["source"]["type"] == "image_data_url"
    assert payload["source"]["path"] is None
    assert payload["source"]["origin"] == {**source_context, "video_path": "[redacted_path]"}
    assert payload["vision"]["source"]["type"] == "image_data_url"
    assert payload["vision"]["source"]["path"] is None
    assert payload["vision"]["source"]["origin"] == {**source_context, "video_path": "[redacted_path]"}


def test_analyze_frame_data_url_reports_decode_failure() -> None:
    payload = analyze_frame_data_url("tft", "data:image/png;base64,not-base64")

    assert payload["success"] is False
    assert payload["error"]["code"] == "image_decode_failed"


def test_realtime_session_debounces_stable_result() -> None:
    session = RealtimeInsightSession()
    session.configure(enabled=True, interval_seconds=2, debounce_seconds=0)
    result = {
        "success": True,
        "state": {"stage": "3-2", "level": 6, "gold": 42},
        "insights": [{"type": "trait_gap", "title": "Mage nearby"}],
    }

    payload = session.ingest(result)

    assert payload["enabled"] is True
    assert payload["frame_count"] == 1
    assert payload["stable_result"] == result


def test_realtime_session_tracks_generic_frame_hash_changes() -> None:
    session = RealtimeInsightSession()
    session.configure(enabled=True, profile_id="generic", debounce_seconds=0)
    first = {
        "success": True,
        "profile": "generic",
        "state": {},
        "insights": [],
        "vision": {
            "frame": {"content_hash": "sha256:first"},
            "scene": {"label": "unknown", "confidence": 0.0},
        },
    }
    repeated = {
        "success": True,
        "profile": "generic",
        "state": {},
        "insights": [],
        "vision": {
            "frame": {"content_hash": "sha256:first"},
            "scene": {"label": "unknown", "confidence": 0.0},
        },
    }
    changed = {
        "success": True,
        "profile": "generic",
        "state": {},
        "insights": [],
        "vision": {
            "frame": {"content_hash": "sha256:second"},
            "scene": {"label": "unknown", "confidence": 0.0},
        },
    }

    first_payload = session.ingest(first)
    repeated_payload = session.ingest(repeated)
    changed_payload = session.ingest(changed)

    assert first_payload["last_frame_changed"] is True
    assert first_payload["repeated_frame_count"] == 1
    assert repeated_payload["last_frame_changed"] is False
    assert repeated_payload["repeated_frame_count"] == 2
    assert changed_payload["last_frame_changed"] is True
    assert changed_payload["repeated_frame_count"] == 1
    assert changed_payload["last_content_hash"] == "sha256:second"
    assert changed_payload["stable_result"] == changed


def test_replay_snapshots_and_training_prompt_roundtrip() -> None:
    store = _Store()
    analysis = {
        "profile": "tft",
        "state": {"level": 6, "gold": 42},
        "insights": [{"type": "trait_gap", "title": "Mage nearby", "detail": "", "related_units": []}],
    }

    snapshot = build_snapshot(analysis, note="round 3-2")
    snapshots = append_snapshot(store, snapshot)
    prompt = build_training_prompt(snapshots[-1])

    assert load_snapshots(store) == snapshots
    assert prompt["type"] == "tft_training_prompt"
    assert prompt["state"]["level"] == 6
    clear_snapshots(store)
    assert load_snapshots(store) == []


def test_replay_snapshot_keeps_source_digest_without_local_path() -> None:
    analysis = {
        "profile": "tft",
        "source": {
            "type": "image_path",
            "path": "C:/captures/tft.png",
            "width": 1920,
            "height": 1080,
            "content_hash": "sha256:test",
        },
        "state": {"level": 6, "gold": 42},
        "insights": [],
    }

    snapshot = build_snapshot(analysis, note="see C:/captures/tft.png")

    assert snapshot["source"] == {
        "type": "image_path",
        "width": 1920,
        "height": 1080,
        "content_hash": "sha256:test",
    }
    assert "C:/captures/tft.png" not in repr(snapshot)
    assert snapshot["note"] == "see [redacted_path]"


def test_replay_snapshot_uses_vision_frame_hash_when_source_omits_it() -> None:
    analysis = {
        "profile": "generic",
        "source": {"type": "image_path", "width": 640, "height": 360},
        "vision": {"frame": {"content_hash": "sha256:vision-frame"}},
        "state": {},
        "insights": [],
    }

    snapshot = build_snapshot(analysis)

    assert snapshot["source"] == {
        "type": "image_path",
        "width": 640,
        "height": 360,
        "content_hash": "sha256:vision-frame",
    }


def test_neko_context_packet_is_queued_summary_only() -> None:
    analysis = {
        "profile": "tft",
        "state": {"stage": "3-2", "level": 6, "gold": 42},
        "source": {"type": "image_path", "path": "C:/captures/tft.png", "width": 1920, "height": 1080},
        "vision": {
            "schema_version": 1,
            "privacy": {"external_model_calls": False, "stores_raw_image": False},
            "diagnostics": {"vlm_fallback": {"status": "skipped", "reason": "not_needed"}},
        },
        "insights": [
            {"type": "trait_gap", "title": "Mage nearby", "detail": "One more unit enables the next tier."},
            {"type": "item_direction", "title": "AP items", "detail": "Rod and Tear point toward caster items."},
        ],
    }

    packet = build_neko_context_packet(analysis, note="manual review")

    assert packet["type"] == "game_companion_neko_context_packet"
    assert packet["schema_version"] == 1
    assert packet["profile"] == "tft"
    assert packet["delivery"]["mode"] == "queued_non_interrupting"
    assert packet["yui_boundary"]["agent_mode"] == "assistance_system"
    assert packet["yui_boundary"]["no_roleplay_identity"] is True
    assert packet["memory_policy"] == "summary_only"
    assert packet["privacy"]["raw_image_included"] is False
    assert packet["privacy"]["external_model_calls"] is False
    assert "source" not in packet
    assert "C:/captures/tft.png" not in repr(packet)
    assert packet["state_digest"] == {"stage": "3-2", "level": 6, "gold": 42}
    assert packet["summary"]
    assert packet["events"][0]["type"] == "trait_gap"
    assert packet["note"] == "manual review"


def test_neko_context_packet_redacts_paths_and_data_urls_from_allowed_strings() -> None:
    analysis = {
        "profile": "tft",
        "state": {"stage": "3-2", "level": 6, "gold": 42},
        "insights": [
            {
                "type": "observation",
                "title": "Path C:/captures/tft.png",
                "detail": "raw data:image/png;base64,AAAA local D:/videos/match.mp4",
            }
        ],
    }

    packet = build_neko_context_packet(analysis, note="see C:/captures/tft.png and data:image/png;base64,BBBB")

    serialized = repr(packet)
    assert "C:/captures/tft.png" not in serialized
    assert "D:/videos/match.mp4" not in serialized
    assert "data:image/png;base64" not in serialized
    assert "[redacted_path]" in serialized
    assert "[redacted_image_data]" in serialized


def test_neko_context_queue_is_fifo_and_sanitized() -> None:
    store = _Store()
    analysis = {
        "profile": "tft",
        "state": {"stage": "3-2", "level": 6, "gold": 42},
        "source": {"type": "image_path", "path": "C:/captures/tft.png", "width": 1920, "height": 1080},
        "vision": {
            "schema_version": 1,
            "privacy": {"external_model_calls": False, "stores_raw_image": False},
            "diagnostics": {
                "ocr": {
                    "regions": {
                        "stage": {
                            "text": "SECRET OCR TEXT",
                            "boxes": [[1, 2, 3, 4]],
                            "bbox": [1, 2, 3, 4],
                        }
                    }
                }
            },
        },
        "insights": [{"type": "trait_gap", "title": "Mage nearby", "detail": "One more unit enables the next tier."}],
    }
    first = build_neko_context_packet(analysis, note="first")
    second = build_neko_context_packet({**analysis, "state": {"stage": "3-3"}}, note="second")

    first_result = enqueue_neko_context_packet(store, first)
    second_result = enqueue_neko_context_packet(store, second)
    queued = list_neko_context_queue(store)
    dequeued = dequeue_neko_context_packet(store)

    assert first_result["queued"] is True
    assert first_result["queue_size"] == 1
    assert second_result["queue_size"] == 2
    assert len(queued) == 2
    assert dequeued["available"] is True
    assert dequeued["packet"]["note"] == "first"
    assert list_neko_context_queue(store)[0]["note"] == "second"
    assert "C:/captures/tft.png" not in repr(queued)
    assert "SECRET OCR TEXT" not in repr(queued)
    assert "[1, 2, 3, 4]" not in repr(queued)


def test_neko_context_queue_redacts_adversarial_packet_strings() -> None:
    store = _Store()
    packet = {
        "type": "game_companion_neko_context_packet",
        "schema_version": 1,
        "profile": "tft",
        "events": [
            {
                "type": "observation",
                "title": "C:/captures/tft.png",
                "detail": "data:image/png;base64,AAAA",
                "confidence": 0.9,
            }
        ],
        "summary": "D:/videos/match.mp4",
        "note": "data:image/jpeg;base64,BBBB",
    }

    enqueue_neko_context_packet(store, packet)
    queued = list_neko_context_queue(store)
    dequeued = dequeue_neko_context_packet(store)

    serialized = repr({"queued": queued, "dequeued": dequeued})
    assert "C:/captures/tft.png" not in serialized
    assert "D:/videos/match.mp4" not in serialized
    assert "data:image/png;base64" not in serialized
    assert "data:image/jpeg;base64" not in serialized
    assert "[redacted_path]" in serialized
    assert "[redacted_image_data]" in serialized
