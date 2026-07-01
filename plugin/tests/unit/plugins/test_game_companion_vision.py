from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from plugin.plugins.game_companion.core.frame_analyzer import analyze_frame
from plugin.plugins.game_companion.core.local_vision import (
    LocalVisionBackend,
    analyze_local_vision,
    analyze_local_vision_async,
    reset_default_local_vision_backend,
    set_default_local_vision_backend,
)
from plugin.plugins.game_companion.core.onnx_local_vision import (
    OnnxClassifierConfig,
    create_onnx_classifier_backend,
    load_onnx_classifier_config,
)
from plugin.plugins.game_companion.core.vlm_fallback import (
    VlmFallbackPolicy,
    apply_vlm_fallback_plan,
    build_vlm_fallback_plan,
)
from plugin.plugins.game_companion.core.vlm_input import prepare_vlm_input
from plugin.plugins.game_companion.core.vlm_provider import (
    DisabledVlmProvider,
    VlmProviderResult,
    apply_vlm_provider_result,
    run_vlm_provider,
)
from plugin.plugins.game_companion.core.vision_schema import VisionFrameAnalysis


def test_vision_frame_analysis_serializes_required_contract() -> None:
    payload = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": "sample.png", "width": 640, "height": 360},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "unknown", "confidence": 0.0},
        game_state={"stage": None},
        diagnostics={"warnings": []},
    ).to_dict()

    assert payload["schema_version"] == 1
    assert payload["profile_id"] == "generic"
    assert payload["source"]["width"] == 640
    assert payload["frame"]["content_hash"] == "sha256:test"
    assert payload["scene"] == {"label": "unknown", "confidence": 0.0}
    assert payload["text"] == []
    assert payload["objects"] == []
    assert payload["ui"] == []
    assert payload["game_state"] == {"stage": None}
    assert payload["insights"] == []
    assert payload["suggestions"] == []
    assert payload["confidence"] == 0.0
    assert payload["privacy"]["stores_raw_image"] is False
    assert payload["model_calls"] == []


def test_vision_frame_analysis_keeps_privacy_flags_read_only() -> None:
    payload = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": "sample.png"},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "unknown", "confidence": 0.0},
        privacy={
            "stores_raw_image": True,
            "external_model_calls": True,
            "custom_policy": "kept",
        },
        model_calls=[{"provider": "external"}],
    ).to_dict()

    assert payload["privacy"]["stores_raw_image"] is False
    assert payload["privacy"]["external_model_calls"] is False
    assert payload["privacy"]["custom_policy"] == "kept"
    assert payload["model_calls"] == []


def test_source_context_sanitizes_raw_frame_content() -> None:
    from plugin.plugins.game_companion.core.vision_schema import source_with_origin

    source = {"type": "image_data_url", "path": None, "width": 640, "height": 360}
    enriched = source_with_origin(
        source,
        {
            "type": "video_frame",
            "profile_id": "generic",
            "video_path": "D:/captures/match.mp4",
            "frame_index": 120,
            "timestamp_seconds": 12.0,
            "image_data_url": "data:image/png;base64,AAAA",
            "data_base64": "AAAA",
            "raw_frame": {"pixels": "secret"},
        },
    )

    assert enriched["origin"] == {
        "type": "video_frame",
        "profile_id": "generic",
        "video_path": "[redacted_path]",
        "frame_index": 120,
        "timestamp_seconds": 12.0,
    }


def test_analyze_frame_omits_local_paths_from_normal_json(tmp_path: Path) -> None:
    screenshot = tmp_path / "generic_video_frame.png"
    video_path = tmp_path / "match.mp4"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    payload = analyze_frame(
        "generic",
        screenshot,
        source_context={
            "type": "video_frame",
            "profile_id": "generic",
            "video_path": str(video_path.resolve()),
            "label": f"sample from {screenshot.resolve()}",
            "note": "raw path data:image/png;base64,AAAA",
            "frame_index": 120,
            "timestamp_seconds": 12.5,
        },
    )

    serialized = str(payload)

    assert payload["source"]["type"] == "image_path"
    assert "path" not in payload["source"]
    assert "path" not in payload["vision"]["source"]
    assert str(screenshot.resolve()) not in serialized
    assert str(video_path.resolve()) not in serialized
    assert "data:image/png;base64" not in serialized
    assert payload["source"]["origin"]["video_path"] == "[redacted_path]"


def test_analyze_frame_errors_redact_local_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing_frame.png"

    generic_payload = analyze_frame("generic", missing)
    tft_payload = analyze_frame("tft", missing)

    for payload in (generic_payload, tft_payload):
        serialized = str(payload)
        assert payload["success"] is False
        assert payload["error"]["code"] == "image_not_found"
        assert str(missing) not in serialized
        assert "[redacted_path]" in serialized or payload["error"]["message"] == "image file was not found"


def test_vlm_fallback_policy_plans_low_confidence_without_external_call() -> None:
    vision = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": "sample.png", "width": 640, "height": 360},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "unknown", "confidence": 0.1},
        confidence=0.1,
    ).to_dict()

    plan = build_vlm_fallback_plan(vision, policy=VlmFallbackPolicy(confidence_threshold=0.5))
    merged = apply_vlm_fallback_plan(vision, plan)

    assert plan["status"] == "planned"
    assert plan["reason"] == "low_confidence"
    assert plan["model_role"] == "vision"
    assert plan["max_tokens"] == 500
    assert plan["requires_desensitization"] is True
    assert plan["send_full_frame"] is False
    assert plan["input_policy"]["preferred_payload"] == "cropped_regions"
    assert plan["input_policy"]["send_full_frame"] is False
    assert plan["input_policy"]["requires_desensitization"] is True
    assert "chat_area" in plan["input_policy"]["redact_regions"]
    assert "player_names" in plan["input_policy"]["redact_regions"]
    assert merged["diagnostics"]["vlm_fallback"] == plan
    assert merged["privacy"]["external_model_calls"] is False
    assert merged["privacy"]["requires_desensitization"] is True
    assert merged["privacy"]["vlm_input_policy"] == plan["input_policy"]
    assert merged["model_calls"] == []


def test_vlm_fallback_policy_skips_confident_frame_unless_requested() -> None:
    vision = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": "sample.png", "width": 640, "height": 360},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "menu", "confidence": 0.92},
        confidence=0.92,
    ).to_dict()

    skipped = build_vlm_fallback_plan(vision, policy=VlmFallbackPolicy(confidence_threshold=0.5))
    requested = build_vlm_fallback_plan(
        vision,
        user_requested=True,
        policy=VlmFallbackPolicy(confidence_threshold=0.5),
    )

    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "not_needed"
    assert requested["status"] == "planned"
    assert requested["reason"] == "user_requested"


def test_vlm_fallback_policy_uses_scene_confidence_when_top_level_is_default() -> None:
    vision = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": "sample.png", "width": 640, "height": 360},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "shop", "confidence": 0.88},
    ).to_dict()

    plan = build_vlm_fallback_plan(vision, policy=VlmFallbackPolicy(confidence_threshold=0.5))

    assert vision["confidence"] == 0.0
    assert plan["status"] == "skipped"
    assert plan["reason"] == "not_needed"


def test_vlm_fallback_policy_plans_for_unknown_ui_even_when_confident() -> None:
    vision = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": "sample.png", "width": 640, "height": 360},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "menu", "confidence": 0.91},
        ui=[
            {
                "type": "popup_dialog",
                "label": "unknown",
                "bbox": [120, 80, 520, 280],
                "confidence": 0.86,
            }
        ],
        confidence=0.91,
    ).to_dict()

    plan = build_vlm_fallback_plan(vision, policy=VlmFallbackPolicy(confidence_threshold=0.5))

    assert plan["status"] == "planned"
    assert plan["reason"] == "unknown_ui"
    assert plan["external_call_executed"] is False


def test_prepare_vlm_input_crops_detected_ui_and_redacts_sensitive_bboxes(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    image = Image.new("RGB", (100, 100), color=(240, 240, 240))
    for x in range(30, 45):
        for y in range(30, 45):
            image.putpixel((x, y), (255, 0, 0))
    image.save(screenshot)
    vision = VisionFrameAnalysis(
        profile_id="tft",
        source={"type": "image_path", "path": str(screenshot), "width": 100, "height": 100},
        frame={"width": 100, "height": 100, "content_hash": "sha256:test"},
        scene={"label": "tft_unknown", "confidence": 0.4},
        ui=[
            {
                "type": "overlay",
                "label": "unknown",
                "bbox": [20, 20, 80, 80],
                "confidence": 0.8,
            }
        ],
        privacy={"redact_bboxes": [[30, 30, 45, 45]]},
        confidence=0.4,
    ).to_dict()
    plan = build_vlm_fallback_plan(vision)

    prepared = prepare_vlm_input(vision, screenshot, plan=plan)

    assert prepared["status"] == "prepared"
    assert prepared["external_call_executed"] is False
    assert prepared["privacy"]["raw_image_logging"] is False
    assert prepared["privacy"]["contains_full_frame"] is False
    assert prepared["payload_kind"] == "cropped_regions"
    assert len(prepared["payloads"]) == 1
    assert prepared["payloads"][0]["bbox"] == [20, 20, 80, 80]
    payload_bytes = base64.b64decode(prepared["payloads"][0]["data_base64"])
    cropped = Image.open(BytesIO(payload_bytes))
    assert cropped.size == (60, 60)
    assert cropped.getpixel((10, 10)) == (0, 0, 0)


def test_prepare_vlm_input_blocks_planned_frame_without_safe_regions(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(240, 240, 240)).save(screenshot)
    vision = VisionFrameAnalysis(
        profile_id="tft",
        source={"type": "image_path", "path": str(screenshot), "width": 100, "height": 100},
        frame={"width": 100, "height": 100, "content_hash": "sha256:test"},
        scene={"label": "tft_unknown", "confidence": 0.1},
        confidence=0.1,
    ).to_dict()
    plan = build_vlm_fallback_plan(vision)

    prepared = prepare_vlm_input(vision, screenshot, plan=plan)

    assert prepared["status"] == "blocked"
    assert prepared["reason"] == "no_safe_regions"
    assert prepared["payloads"] == []
    assert prepared["external_call_executed"] is False


def test_prepare_vlm_input_skips_when_fallback_is_not_planned(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(240, 240, 240)).save(screenshot)
    vision = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "path": str(screenshot), "width": 100, "height": 100},
        frame={"width": 100, "height": 100, "content_hash": "sha256:test"},
        scene={"label": "menu", "confidence": 0.95},
        confidence=0.95,
    ).to_dict()
    plan = build_vlm_fallback_plan(vision)

    prepared = prepare_vlm_input(vision, screenshot, plan=plan)

    assert prepared["status"] == "skipped"
    assert prepared["reason"] == "not_needed"
    assert prepared["payloads"] == []


def test_prepare_vlm_input_blocks_full_frame_without_redaction_proof(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(240, 240, 240)).save(screenshot)
    vision = VisionFrameAnalysis(
        profile_id="tft",
        source={"type": "image_path", "path": str(screenshot), "width": 100, "height": 100},
        frame={"width": 100, "height": 100, "content_hash": "sha256:test"},
        scene={"label": "tft_unknown", "confidence": 0.1},
        confidence=0.1,
    ).to_dict()
    plan = {
        "status": "planned",
        "reason": "user_requested",
        "send_full_frame": True,
        "requires_desensitization": True,
        "input_policy": {
            "send_full_frame": True,
            "preferred_payload": "desensitized_frame",
            "requires_desensitization": True,
        },
    }

    prepared = prepare_vlm_input(vision, screenshot, plan=plan)

    assert prepared["status"] == "blocked"
    assert prepared["reason"] == "missing_desensitization_redactions"


def test_prepare_vlm_input_blocks_type_d_crops_without_redaction_proof(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(20, 30, 40)).save(screenshot)
    vision = VisionFrameAnalysis(
        profile_id="tft",
        source={"type": "image_path", "width": 1920, "height": 1080},
        frame={"width": 1920, "height": 1080, "content_hash": "sha256:test"},
        scene={"label": "tft_unknown", "confidence": 0.1},
        ui=[{"type": "dialog", "label": "augment", "bbox": [100, 100, 500, 300]}],
        confidence=0.1,
    ).to_dict()
    plan = build_vlm_fallback_plan(vision)

    prepared = prepare_vlm_input(vision, screenshot, plan=plan)

    assert prepared["status"] == "blocked"
    assert prepared["reason"] == "missing_desensitization_redactions"
    assert prepared["payloads"] == []


def test_prepare_vlm_input_blocks_sensitive_ui_regions(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (100, 100), color=(240, 240, 240)).save(screenshot)
    vision = VisionFrameAnalysis(
        profile_id="tft",
        source={"type": "image_path", "path": str(screenshot), "width": 100, "height": 100},
        frame={"width": 100, "height": 100, "content_hash": "sha256:test"},
        scene={"label": "tft_unknown", "confidence": 0.2},
        ui=[
            {
                "type": "chat_area",
                "label": "player_names",
                "bbox": [10, 10, 80, 40],
                "confidence": 0.9,
            }
        ],
        confidence=0.2,
    ).to_dict()
    plan = build_vlm_fallback_plan(vision)

    prepared = prepare_vlm_input(vision, screenshot, plan=plan)

    assert prepared["status"] == "blocked"
    assert prepared["reason"] == "no_safe_regions"


def test_disabled_vlm_provider_never_executes_external_call() -> None:
    preparation = {
        "type": "vlm_input_preparation",
        "status": "prepared",
        "payloads": [{"data_base64": "SECRET"}],
        "payload_kind": "cropped_regions",
    }

    result = run_vlm_provider(DisabledVlmProvider(), preparation)

    assert result.status == "skipped"
    assert result.reason == "provider_disabled"
    assert result.external_call_executed is False
    assert result.model_calls == []


def test_vlm_provider_result_merges_without_raw_payload_leak() -> None:
    vision = VisionFrameAnalysis(
        profile_id="generic",
        source={"type": "image_path", "width": 640, "height": 360},
        frame={"width": 640, "height": 360, "content_hash": "sha256:test"},
        scene={"label": "unknown", "confidence": 0.2},
        confidence=0.2,
    ).to_dict()
    result = VlmProviderResult(
        status="merged",
        reason="provider_completed",
        scene={"label": "menu", "confidence": 0.77},
        insights=[{"type": "vlm_summary", "title": "Menu visible", "detail": "Start button is visible."}],
        suggestions=["inspect menu options"],
        model_calls=[{"provider": "fake_vlm", "model": "fake-vision"}],
        external_call_executed=True,
        raw_payload={"data_base64": "SECRET", "path": "C:/captures/frame.png"},
    )

    merged = apply_vlm_provider_result(vision, result)
    serialized = repr(merged)

    assert merged["scene"]["label"] == "menu"
    assert merged["scene"]["confidence"] == 0.77
    assert merged["insights"][0]["type"] == "vlm_summary"
    assert merged["suggestions"] == ["inspect menu options"]
    assert merged["diagnostics"]["vlm_provider"]["status"] == "merged"
    assert merged["privacy"]["external_model_calls"] is True
    assert merged["model_calls"] == [{"provider": "fake_vlm", "model": "fake-vision"}]
    assert "SECRET" not in serialized
    assert "C:/captures/frame.png" not in serialized
    assert "raw_payload" not in serialized


def test_local_vision_analyzer_sanitizes_backend_paths_and_exception_text(tmp_path: Path) -> None:
    screenshot = tmp_path / "generic.png"
    leaked_path = str((tmp_path / "model.onnx").resolve())
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    def classifier(_path: Path, _profile_id: str) -> dict[str, object]:
        return {"label": "menu", "confidence": 0.9, "model_name": leaked_path, "note": "data:image/png;base64,AAAA"}

    def detector(_path: Path, _profile_id: str) -> dict[str, object]:
        raise RuntimeError(f"failed reading {leaked_path}")

    payload = analyze_local_vision(
        screenshot,
        profile_id="generic",
        backend=LocalVisionBackend(classifier=classifier, detector=detector),
    )
    serialized = str(payload)

    assert leaked_path not in serialized
    assert "data:image/png;base64" not in serialized
    assert payload["scene"]["model_name"] == "[redacted_path]"
    assert payload["diagnostics"]["detector"]["error"] == "failed reading [redacted_path]"


def test_local_vision_analyzer_skips_without_backends(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    payload = analyze_local_vision(screenshot, profile_id="generic")

    assert payload["available"] is False
    assert payload["scene"] == {"label": "unknown", "confidence": 0.0}
    assert payload["objects"] == []
    assert payload["ui"] == []
    assert payload["diagnostics"]["classifier"]["status"] == "skipped"
    assert payload["diagnostics"]["detector"]["status"] == "skipped"


def test_local_vision_analyzer_merges_classifier_and_detector(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    backend = LocalVisionBackend(
        classifier=lambda _path, _profile_id: {
            "label": "menu",
            "confidence": 0.82,
            "all_scores": {"menu": 0.82, "combat": 0.18},
            "latency_ms": 3.5,
            "model_name": "demo_classifier",
        },
        detector=lambda _path, _profile_id: {
            "objects": [{"label": "button", "bbox": [10, 20, 110, 60], "confidence": 0.9}],
            "ui": [{"type": "button", "label": "Start", "bbox": [10, 20, 110, 60], "confidence": 0.9}],
        },
    )

    payload = analyze_local_vision(screenshot, profile_id="generic", backend=backend)

    assert payload["available"] is True
    assert payload["scene"]["label"] == "menu"
    assert payload["scene"]["confidence"] == 0.82
    assert payload["scene"]["all_scores"]["combat"] == 0.18
    assert payload["scene"]["latency_ms"] == 3.5
    assert payload["scene"]["model_name"] == "demo_classifier"
    assert payload["objects"][0]["label"] == "button"
    assert payload["ui"][0]["type"] == "button"
    assert payload["diagnostics"]["classifier"]["status"] == "ready"
    assert payload["diagnostics"]["detector"]["status"] == "ready"


@pytest.mark.asyncio
async def test_local_vision_analyzer_async_wrapper(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    backend = LocalVisionBackend(classifier=lambda _path, _profile_id: {"label": "combat", "confidence": 0.7})

    payload = await analyze_local_vision_async(screenshot, profile_id="generic", backend=backend)

    assert payload["scene"]["label"] == "combat"
    assert payload["diagnostics"]["classifier"]["status"] == "ready"


def test_local_vision_analyzer_marks_backend_errors_failed(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    backend = LocalVisionBackend(classifier=lambda _path, _profile_id: {"error": "model missing"})

    payload = analyze_local_vision(screenshot, profile_id="generic", backend=backend)

    assert payload["diagnostics"]["classifier"] == {"status": "failed", "error": "model missing"}
    assert payload["scene"]["label"] == "unknown"
    assert payload["scene"]["confidence"] == 0.0


def test_local_vision_analyzer_captures_backend_exceptions(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    def broken_classifier(_path: Path, _profile_id: str) -> dict[str, object]:
        raise RuntimeError("onnx session crashed")

    backend = LocalVisionBackend(classifier=broken_classifier)

    payload = analyze_local_vision(screenshot, profile_id="generic", backend=backend)

    assert payload["available"] is False
    assert payload["scene"] == {"label": "unknown", "confidence": 0.0}
    assert payload["diagnostics"]["classifier"]["status"] == "failed"
    assert payload["diagnostics"]["classifier"]["error"] == "onnx session crashed"
    assert payload["diagnostics"]["classifier"]["error_type"] == "RuntimeError"


def test_local_vision_analyzer_rejects_invalid_backend_payloads(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    backend = LocalVisionBackend(
        classifier=lambda _path, _profile_id: {},
        detector=lambda _path, _profile_id: {"objects": "not-a-list"},
    )

    payload = analyze_local_vision(screenshot, profile_id="generic", backend=backend)

    assert payload["available"] is False
    assert payload["scene"] == {"label": "unknown", "confidence": 0.0}
    assert payload["objects"] == []
    assert payload["ui"] == []
    assert payload["diagnostics"]["classifier"]["status"] == "failed"
    assert payload["diagnostics"]["classifier"]["reason"] == "invalid_payload"
    assert payload["diagnostics"]["detector"]["status"] == "failed"
    assert payload["diagnostics"]["detector"]["reason"] == "invalid_payload"


def test_local_vision_analyzer_sanitizes_backend_raw_content(tmp_path: Path) -> None:
    screenshot = tmp_path / "frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    backend = LocalVisionBackend(
        classifier=lambda _path, _profile_id: {
            "label": "menu",
            "confidence": 0.8,
            "image_data_url": "data:image/png;base64,AAAA",
            "model_path": "C:/models/classifier.onnx",
        },
        detector=lambda _path, _profile_id: {
            "ui": [
                {
                    "type": "button",
                    "label": "Start",
                    "bbox": [10, 20, 110, 60],
                    "confidence": 0.9,
                    "data_base64": "AAAA",
                    "crop_path": "C:/captures/crop.png",
                }
            ],
        },
    )

    payload = analyze_local_vision(screenshot, profile_id="generic", backend=backend)

    serialized = repr(payload)
    assert payload["scene"]["label"] == "menu"
    assert payload["ui"][0]["label"] == "Start"
    assert "data:image/png;base64" not in serialized
    assert "C:/models/classifier.onnx" not in serialized
    assert "C:/captures/crop.png" not in serialized
    assert "data_base64" not in serialized
    assert "crop_path" not in serialized


def test_default_local_vision_backend_powers_generic_analyzer(tmp_path: Path) -> None:
    screenshot = tmp_path / "generic.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    backend = LocalVisionBackend(
        classifier=lambda _path, _profile_id: {"label": "combat", "confidence": 0.73},
        detector=lambda _path, _profile_id: {
            "ui": [{"type": "status_bar", "label": "HP", "bbox": [10, 10, 90, 20], "confidence": 0.81}],
            "objects": [{"label": "health_bar", "bbox": [10, 10, 90, 20], "confidence": 0.81}],
        },
    )

    set_default_local_vision_backend(backend)
    try:
        payload = analyze_frame("generic", screenshot)
    finally:
        reset_default_local_vision_backend()

    assert payload["vision"]["scene"]["label"] == "combat"
    assert payload["vision"]["scene"]["confidence"] == 0.73
    assert payload["vision"]["objects"][0]["label"] == "health_bar"
    assert payload["vision"]["ui"][0]["type"] == "status_bar"
    assert payload["vision"]["diagnostics"]["analyzers"]["classifier"]["status"] == "ready"
    assert payload["vision"]["diagnostics"]["analyzers"]["detector"]["status"] == "ready"


def test_onnx_classifier_backend_feeds_local_vision_scene(tmp_path: Path) -> None:
    class _Input:
        name = "pixels"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def run(self, _output_names, _feeds):
            return [[[0.05, 2.4, 0.2]]]

    screenshot = tmp_path / "frame.png"
    model_path = tmp_path / "screen_classifier.onnx"
    Image.new("RGB", (80, 60), color=(120, 90, 80)).save(screenshot)
    model_path.write_bytes(b"fake")
    backend = create_onnx_classifier_backend(
        OnnxClassifierConfig(
            model_path=model_path,
            labels=("loading", "shop", "combat"),
            model_name="test-screen-classifier",
            input_size=(32, 32),
        ),
        session_factory=lambda _path: _Session(),
    )

    payload = analyze_local_vision(
        screenshot,
        profile_id="tft",
        backend=LocalVisionBackend(classifier=backend),
    )

    assert payload["available"] is True
    assert payload["scene"]["label"] == "shop"
    assert payload["scene"]["confidence"] > 0.8
    assert payload["scene"]["model_name"] == "test-screen-classifier"
    assert payload["scene"]["all_scores"]["shop"] > payload["scene"]["all_scores"]["combat"]
    assert payload["diagnostics"]["classifier"]["status"] == "ready"


def test_onnx_classifier_backend_reports_label_mismatch_without_path_leak(tmp_path: Path) -> None:
    class _Input:
        name = "pixels"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def run(self, _output_names, _feeds):
            return [[[0.2, 0.8, 0.1]]]

    screenshot = tmp_path / "frame.png"
    model_path = tmp_path / "screen_classifier.onnx"
    Image.new("RGB", (80, 60), color=(120, 90, 80)).save(screenshot)
    model_path.write_bytes(b"fake")
    backend = create_onnx_classifier_backend(
        OnnxClassifierConfig(
            model_path=model_path,
            labels=("shop", "combat"),
            model_name="test-screen-classifier",
        ),
        session_factory=lambda _path: _Session(),
    )

    payload = analyze_local_vision(
        screenshot,
        profile_id="tft",
        backend=LocalVisionBackend(classifier=backend),
    )

    assert payload["available"] is False
    assert payload["diagnostics"]["classifier"]["status"] == "failed"
    assert "logits_label_mismatch" in payload["diagnostics"]["classifier"]["error"]
    assert str(model_path) not in str(payload)
    assert str(screenshot) not in str(payload)


def test_load_onnx_classifier_config_reads_label_map_without_path_leak(tmp_path: Path) -> None:
    model_path = tmp_path / "screen_classifier.onnx"
    labels_path = tmp_path / "labels.json"
    model_path.write_bytes(b"fake")
    labels_path.write_text('{"labels": ["shop", "combat", "augment_select"]}', encoding="utf-8")

    config = load_onnx_classifier_config(
        {
            "enabled": True,
            "model_path": str(model_path),
            "labels_path": str(labels_path),
            "model_name": "tft-screen-classifier",
            "input_size": [128, 96],
        },
        base_dir=tmp_path,
    )

    assert config is not None
    assert config.model_path == model_path
    assert config.labels == ("shop", "combat", "augment_select")
    assert config.model_name == "tft-screen-classifier"
    assert config.input_size == (128, 96)


def test_load_onnx_classifier_config_returns_none_when_disabled_or_incomplete(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text('{"labels": ["shop"]}', encoding="utf-8")

    assert load_onnx_classifier_config({"enabled": False}, base_dir=tmp_path) is None
    assert load_onnx_classifier_config({"enabled": True, "labels_path": str(labels_path)}, base_dir=tmp_path) is None
    assert load_onnx_classifier_config({"enabled": True, "model_path": "model.onnx"}, base_dir=tmp_path) is None


def test_tft_analyzer_respects_special_layout_hint_without_shop_recognition(tmp_path: Path) -> None:
    screenshot = tmp_path / "special.png"
    Image.new("RGB", (1920, 1080), color=(40, 40, 40)).save(screenshot)

    payload = analyze_frame(
        "tft",
        screenshot,
        source_context={
            "type": "video_frame",
            "profile_id": "tft",
            "expected_layout": "special",
            "frame_index": 0,
            "timestamp_seconds": 0.0,
        },
    )

    assert payload["success"] is True
    assert payload["state"]["shop_units"] == []
    assert payload["diagnostics"]["recognition"]["status"] == "skipped"
    assert payload["diagnostics"]["recognition"]["reason"] == "special_layout"
    assert payload["vision"]["scene"]["label"] == "tft_special"
    assert payload["vision"]["game_state"]["layout"] == "special"
    assert payload["vision"]["diagnostics"]["layout_hint"] == "special"


def test_analyze_frame_supports_generic_unified_vision_json(tmp_path: Path) -> None:
    screenshot = tmp_path / "generic.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    payload = analyze_frame("generic", screenshot)

    assert payload["success"] is True
    assert payload["ok"] is True
    assert payload["profile"] == "generic"
    assert payload["state"] == {}
    assert payload["regions"] == {}
    assert payload["insights"] == []
    assert payload["source"]["type"] == "image_path"
    assert payload["source"]["width"] == 640
    assert payload["source"]["height"] == 360

    vision = payload["vision"]
    assert vision["schema_version"] == 1
    assert vision["profile_id"] == "generic"
    assert vision["scene"]["label"] == "unknown"
    assert vision["frame"]["content_hash"].startswith("sha256:")
    assert vision["frame"]["quality"]["status"] == "ok"
    assert vision["frame"]["quality"]["flags"] == []
    assert vision["privacy"]["stores_raw_image"] is False
    assert vision["privacy"]["external_model_calls"] is False
    assert vision["privacy"]["requires_desensitization"] is True
    assert vision["diagnostics"]["analyzers"]["ocr"]["status"] == "skipped"
    assert vision["diagnostics"]["analyzers"]["detector"]["status"] == "skipped"
    assert vision["diagnostics"]["vlm_fallback"]["status"] == "planned"
    assert vision["diagnostics"]["vlm_fallback"]["reason"] == "low_confidence"


def test_analyze_frame_preserves_video_frame_origin_in_source_context(tmp_path: Path) -> None:
    screenshot = tmp_path / "generic_video_frame.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)
    source_context = {
        "type": "video_frame",
        "profile_id": "generic",
        "video_path": str((tmp_path / "match.mp4").resolve()),
        "ordinal": 3,
        "frame_index": 120,
        "timestamp_seconds": 12.5,
    }

    payload = analyze_frame("generic", screenshot, source_context=source_context)

    assert payload["source"]["type"] == "image_path"
    assert payload["source"]["origin"] == {
        **source_context,
        "video_path": "[redacted_path]",
    }
    assert payload["vision"]["source"]["origin"] == payload["source"]["origin"]


def test_generic_analyzer_reports_basic_quality_flags(tmp_path: Path) -> None:
    screenshot = tmp_path / "dark.png"
    Image.new("RGB", (200, 120), color=(0, 0, 0)).save(screenshot)

    payload = analyze_frame("generic", screenshot)

    assert payload["success"] is True
    quality = payload["vision"]["frame"]["quality"]
    assert quality["status"] == "needs_review"
    assert "too_dark" in quality["flags"]
    assert "low_resolution" in quality["flags"]


def test_tft_analysis_exposes_unified_vision_schema(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(0, 0, 0)).save(screenshot)

    payload = analyze_frame("tft", screenshot)

    assert payload["success"] is True
    vision = payload["vision"]
    assert vision["schema_version"] == 1
    assert vision["profile_id"] == "tft"
    assert vision["scene"]["label"] == "tft_unknown"
    assert vision["source"]["width"] == 1920
    assert vision["frame"]["quality"]["status"] == "needs_review"
    assert "too_dark" in vision["frame"]["quality"]["flags"]
    assert vision["game_state"] == payload["state"]
    assert vision["insights"] == payload["insights"]
    assert vision["diagnostics"]["warnings"] == payload["diagnostics"]["warnings"]
    assert vision["privacy"]["stores_raw_image"] is False
    assert vision["privacy"]["external_model_calls"] is False
    assert vision["diagnostics"]["vlm_fallback"]["status"] == "skipped"
    assert vision["diagnostics"]["vlm_fallback"]["reason"] == "not_needed"


def test_tft_analysis_can_plan_vlm_when_user_requested(tmp_path: Path) -> None:
    screenshot = tmp_path / "tft.png"
    Image.new("RGB", (1920, 1080), color=(10, 20, 30)).save(screenshot)

    payload = analyze_frame("tft", screenshot, vlm_requested=True)

    vision = payload["vision"]
    assert vision["diagnostics"]["vlm_fallback"]["status"] == "planned"
    assert vision["diagnostics"]["vlm_fallback"]["reason"] == "user_requested"
    preparation = vision["diagnostics"]["vlm_input_preparation"]
    assert preparation["type"] == "vlm_input_preparation_summary"
    assert preparation["status"] == "blocked"
    assert preparation["reason"] == "missing_desensitization_redactions"
    assert preparation["payload_count"] == 0
    assert "payloads" not in preparation
    assert "data_base64" not in str(vision)
    assert vision["privacy"]["external_model_calls"] is False
    assert vision["model_calls"] == []


def test_generic_vlm_plan_records_input_preparation_summary_without_payload(tmp_path: Path) -> None:
    screenshot = tmp_path / "generic.png"
    Image.new("RGB", (640, 360), color=(90, 100, 110)).save(screenshot)

    payload = analyze_frame("generic", screenshot, vlm_requested=True)

    preparation = payload["vision"]["diagnostics"]["vlm_input_preparation"]
    assert preparation["type"] == "vlm_input_preparation_summary"
    assert preparation["status"] == "blocked"
    assert preparation["reason"] == "no_safe_regions"
    assert preparation["payload_count"] == 0
    assert "payloads" not in preparation
    assert "data_base64" not in str(payload["vision"])
