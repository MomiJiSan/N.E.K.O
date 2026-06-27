from pathlib import Path
import base64

from PIL import Image

from plugin.plugins.game_companion.core.frame_analyzer import analyze_frame_data_url
from plugin.plugins.game_companion.core.realtime import RealtimeInsightSession
from plugin.plugins.game_companion.core.replay import (
    append_snapshot,
    build_snapshot,
    build_training_prompt,
    clear_snapshots,
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
