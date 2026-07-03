from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageDraw

from .calibration import DEFAULT_LOCAL_CALIBRATION_DIR, SUPPORTED_VIDEO_EXTENSIONS, _read_video_frames
from .tft_recognition import recognize_tft_frame
from .tft_state import build_tft_state

FrameReader = Callable[..., list[dict[str, Any]]]
Recognizer = Callable[..., dict[str, Any]]
RUNTIME_REPORT_VERSION = "tft_state_v1"


def build_tft_video_state_report(
    video_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    sample_interval_seconds: float = 2.0,
    max_frames: int = 60,
    expected_layout: str | None = None,
    frame_indices: list[int] | None = None,
    frame_layouts: Mapping[int | str, str] | None = None,
    shop_labels_path: str | Path | None = None,
    frame_reader: FrameReader | None = None,
    recognizer: Recognizer | None = None,
) -> dict[str, Any]:
    video = Path(video_path).expanduser()
    if not video.is_file():
        raise FileNotFoundError(f"TFT runtime video was not found: {video}")
    if video.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported TFT runtime video extension: {video.suffix}")

    output_path = Path(output_dir).expanduser() if output_dir else _default_runtime_output_dir(video)
    frames_dir = output_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "tft_state_v1.jsonl"
    summary_path = output_path / "tft_state_summary_v1.json"
    contact_sheet_path = output_path / "contact_sheet.png"
    frame_limit = max(1, min(240, int(max_frames or 60)))
    selected_indices = _normalize_frame_indices(frame_indices)
    if not selected_indices and frame_reader is None:
        selected_indices = _interval_frame_indices(video, sample_interval_seconds, frame_limit)
    reader = frame_reader or _read_video_frames
    raw_frames = list(reader(video, frame_indices=selected_indices, max_frames=frame_limit))
    if not raw_frames:
        raise ValueError("no frames extracted from TFT runtime video")

    pending_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    active_recognizer = recognizer or recognize_tft_frame
    shop_labels = _load_runtime_shop_labels(shop_labels_path)
    local_shop_cost_hints = _load_local_shop_cost_hints()
    for ordinal, raw_frame in enumerate(raw_frames, start=1):
        image = raw_frame.get("image")
        if not isinstance(image, Image.Image):
            raise OSError(f"video frame #{ordinal} did not decode to a PIL image")
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        frame_index = _int(raw_frame.get("frame_index"), ordinal - 1)
        timestamp = _float_or_none(raw_frame.get("timestamp_seconds"))
        frame_layout = _layout_for_frame(frame_index, expected_layout, frame_layouts)
        image_path = frames_dir / f"frame_{ordinal:04d}_f{frame_index:06d}.png"
        image.save(image_path)
        source_context = {
            "type": "video_frame",
            "profile_id": "tft",
            "video_path": "[redacted_path]",
            "ordinal": ordinal,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
        }
        if frame_layout:
            source_context["expected_layout"] = frame_layout
        recognition = active_recognizer(image_path, expected_layout=frame_layout)
        _apply_runtime_shop_labels(recognition, frame_index, shop_labels)
        pending_records.append(
            {
                "type": "tft_runtime_state_record",
                "schema_version": 1,
                "ordinal": ordinal,
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "image_path": str(image_path.resolve()),
                "source_context": source_context,
                "raw_recognition": recognition,
            }
        )

    _apply_runtime_temporal_shop_costs(
        [
            record["raw_recognition"]
            for record in pending_records
            if isinstance(record.get("raw_recognition"), dict)
        ]
    )
    for pending in pending_records:
        recognition = pending["raw_recognition"]
        _apply_runtime_local_shop_cost_hints(recognition, local_shop_cost_hints)
        _refresh_runtime_shop_readiness(recognition)
        state = build_tft_state(
            recognition,
            timestamp=pending.get("timestamp_seconds"),
            source_context=pending.get("source_context"),
        )
        record = {
            "type": "tft_runtime_state_record",
            "schema_version": 1,
            "ordinal": pending["ordinal"],
            "frame_index": pending["frame_index"],
            "timestamp_seconds": pending["timestamp_seconds"],
            "image_path": pending["image_path"],
            "state": state,
            "raw_recognition": recognition,
        }
        records.append(record)
        frames.append(
            {
                "ordinal": pending["ordinal"],
                "frame_index": pending["frame_index"],
                "timestamp_seconds": pending["timestamp_seconds"],
                "image_path": pending["image_path"],
                "layout": state["layout"],
                "readiness": state["readiness"],
            }
        )

    _write_jsonl(jsonl_path, records)
    summary = _runtime_summary(records)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_contact_sheet(contact_sheet_path, records)
    return {
        "type": "tft_runtime_state_video_report",
        "schema_version": 1,
        "report_version": RUNTIME_REPORT_VERSION,
        "video_path": str(video.resolve()),
        "output_dir": str(output_path.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "jsonl_path": str(jsonl_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "contact_sheet_path": str(contact_sheet_path.resolve()),
        "sample_interval_seconds": float(sample_interval_seconds),
        "max_frames": frame_limit,
        "requested_frame_indices": selected_indices,
        "frame_layouts": {str(key): value for key, value in (frame_layouts or {}).items()},
        "shop_labels_path": str(Path(shop_labels_path).expanduser().resolve()) if shop_labels_path else "",
        "frame_count": len(records),
        "frames": frames,
        "summary": summary,
    }


def _default_runtime_output_dir(video: Path) -> Path:
    return DEFAULT_LOCAL_CALIBRATION_DIR / video.stem / "runtime_state_v1"


def _interval_frame_indices(video: Path, sample_interval_seconds: float, max_frames: int) -> list[int]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0:
        return []
    step = max(1, int(round(max(0.25, float(sample_interval_seconds or 2.0)) * fps)))
    return list(range(0, frame_count, step))[:max_frames]


def _normalize_frame_indices(frame_indices: list[int] | None) -> list[int]:
    if not frame_indices:
        return []
    normalized = []
    for value in frame_indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index >= 0 and index not in normalized:
            normalized.append(index)
    return normalized


def _layout_for_frame(frame_index: int, expected_layout: str | None, frame_layouts: Mapping[int | str, str] | None) -> str | None:
    if isinstance(frame_layouts, Mapping):
        for key in (frame_index, str(frame_index)):
            value = frame_layouts.get(key)
            if value:
                return str(value)
    return expected_layout


def _load_runtime_shop_labels(path: str | Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    label_path = Path(path).expanduser()
    if not label_path.is_file():
        return {}
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    labels: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in payload.get("samples", []):
        if not isinstance(sample, dict):
            continue
        human = sample.get("human") if isinstance(sample.get("human"), dict) else sample.get("human_label")
        if not isinstance(human, dict) or human.get("status") != "verified":
            continue
        try:
            slot = int(sample.get("slot"))
        except (TypeError, ValueError):
            continue
        for key in _runtime_shop_label_keys(sample):
            labels[(key, slot)] = human
    return labels


def _runtime_shop_label_keys(sample: dict[str, Any]) -> list[str]:
    keys = []
    if sample.get("index") is not None:
        keys.append(str(sample.get("index")))
    for field in ("image_path", "crop_path"):
        value = sample.get(field)
        if not value:
            continue
        match = re.search(r"(?:^|[_-])f0*(\d+)(?:[_\.-]|$)", str(value))
        if match:
            keys.append(str(int(match.group(1))))
    result = []
    for key in keys:
        if key not in result:
            result.append(key)
    return result


def _apply_runtime_shop_labels(
    recognition: dict[str, Any],
    frame_index: int,
    labels: dict[tuple[str, int], dict[str, Any]],
) -> None:
    if not labels or recognition.get("layout") != "normal_shop":
        return
    for slot in recognition.get("shop") or []:
        if not isinstance(slot, dict) or slot.get("state") != "occupied":
            continue
        try:
            slot_number = int(slot.get("slot"))
        except (TypeError, ValueError):
            continue
        human = labels.get((str(frame_index), slot_number))
        if not human:
            continue
        if human.get("name") and not (slot.get("name") or slot.get("name_candidate")):
            slot["name"] = human.get("name")
            slot["name_candidate"] = human.get("name")
            slot["name_candidate_source"] = "human_verified_label"
        if human.get("cost") is not None and slot.get("cost_candidate", slot.get("cost")) is None:
            slot["cost"] = human.get("cost")
            slot["cost_candidate"] = human.get("cost")
            slot["cost_candidate_source"] = "human_verified_label"
            slot["cost_inference"] = {"method": "human_verified_label", "confidence": 1.0}


def _apply_runtime_temporal_shop_costs(recognitions: list[dict[str, Any]]) -> None:
    costs_by_slot_name: dict[tuple[int, str], set[int]] = {}
    costs_by_name: dict[str, set[int]] = {}
    for recognition in recognitions:
        if recognition.get("layout") != "normal_shop":
            continue
        for slot in _runtime_occupied_shop_slots(recognition):
            cost = _runtime_slot_cost(slot)
            name = _runtime_slot_name(slot)
            slot_number = _runtime_slot_number(slot)
            if cost is None or not name:
                continue
            costs_by_name.setdefault(name, set()).add(cost)
            if slot_number is not None:
                costs_by_slot_name.setdefault((slot_number, name), set()).add(cost)

    slot_consensus = {
        key: next(iter(costs))
        for key, costs in costs_by_slot_name.items()
        if len(costs) == 1
    }
    name_consensus = {
        name: next(iter(costs))
        for name, costs in costs_by_name.items()
        if len(costs) == 1
    }
    for recognition in recognitions:
        if recognition.get("layout") != "normal_shop":
            continue
        for slot in _runtime_occupied_shop_slots(recognition):
            if _runtime_slot_cost(slot) is not None:
                continue
            name = _runtime_slot_name(slot)
            slot_number = _runtime_slot_number(slot)
            if not name:
                continue
            if slot_number is not None and (slot_number, name) in slot_consensus:
                _set_runtime_slot_cost(
                    slot,
                    slot_consensus[(slot_number, name)],
                    source="runtime_temporal_slot_name_cost_consensus",
                    confidence=0.74,
                    matched_name=name,
                )
            elif name in name_consensus:
                _set_runtime_slot_cost(
                    slot,
                    name_consensus[name],
                    source="runtime_temporal_name_cost_consensus",
                    confidence=0.68,
                    matched_name=name,
                )


def _apply_runtime_local_shop_cost_hints(recognition: dict[str, Any], cost_hints: Mapping[str, int]) -> None:
    if not cost_hints or recognition.get("layout") != "normal_shop":
        return
    for slot in _runtime_occupied_shop_slots(recognition):
        if _runtime_slot_cost(slot) is not None:
            continue
        name = _runtime_slot_name(slot)
        if not name or name not in cost_hints:
            continue
        _set_runtime_slot_cost(
            slot,
            cost_hints[name],
            source="runtime_local_calibration_name_cost",
            confidence=0.72,
            matched_name=name,
        )


def _load_local_shop_cost_hints(root: str | Path | None = None) -> dict[str, int]:
    calibration_root = Path(root).expanduser() if root is not None else DEFAULT_LOCAL_CALIBRATION_DIR
    if not calibration_root.is_dir():
        return {}
    costs_by_name: dict[str, set[int]] = {}
    for labels_path in calibration_root.rglob("recognition_shop_labels_v1.json"):
        try:
            payload = json.loads(labels_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for sample in payload.get("samples", []):
            if not isinstance(sample, dict):
                continue
            human = sample.get("human") if isinstance(sample.get("human"), dict) else sample.get("human_label")
            if not isinstance(human, dict) or human.get("status") != "verified":
                continue
            name = _normalize_runtime_shop_name(human.get("name"))
            cost = _runtime_valid_cost(human.get("cost"))
            if name and cost is not None:
                costs_by_name.setdefault(name, set()).add(cost)
    return {name: next(iter(costs)) for name, costs in costs_by_name.items() if len(costs) == 1}


def _runtime_occupied_shop_slots(recognition: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        slot
        for slot in recognition.get("shop") or []
        if isinstance(slot, dict) and slot.get("state") == "occupied"
    ]


def _runtime_slot_name(slot: Mapping[str, Any]) -> str:
    return _normalize_runtime_shop_name(slot.get("name_candidate") or slot.get("name"))


def _normalize_runtime_shop_name(value: Any) -> str:
    return str(value or "").strip()


def _runtime_slot_number(slot: Mapping[str, Any]) -> int | None:
    try:
        value = int(slot.get("slot"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _runtime_slot_cost(slot: Mapping[str, Any]) -> int | None:
    cost = _runtime_valid_cost(slot.get("cost_candidate"))
    if cost is not None:
        return cost
    return _runtime_valid_cost(slot.get("cost"))


def _runtime_valid_cost(value: Any) -> int | None:
    try:
        cost = int(value)
    except (TypeError, ValueError):
        return None
    return cost if 1 <= cost <= 5 else None


def _set_runtime_slot_cost(
    slot: dict[str, Any],
    cost: int,
    *,
    source: str,
    confidence: float,
    matched_name: str,
) -> None:
    slot["cost"] = cost
    slot["cost_candidate"] = cost
    slot["cost_candidate_source"] = source
    slot["cost_inference"] = {
        "method": source,
        "matched_name": matched_name,
        "confidence": confidence,
    }


def _refresh_runtime_shop_readiness(recognition: dict[str, Any]) -> None:
    if recognition.get("layout") != "normal_shop" or not recognition.get("success"):
        return
    readiness = recognition.get("readiness") if isinstance(recognition.get("readiness"), dict) else {}
    issues = [
        issue
        for issue in readiness.get("blocking_issues", [])
        if isinstance(issue, dict) and not _runtime_shop_issue_resolved(issue, recognition)
    ]
    if not issues:
        status = "ready"
    elif _has_runtime_shop_evidence(recognition):
        status = "partial"
    else:
        status = "blocked"
    recognition["readiness"] = {**readiness, "readiness": status, "status": status, "blocking_issues": issues}


def _runtime_shop_issue_resolved(issue: dict[str, Any], recognition: dict[str, Any]) -> bool:
    slots = issue.get("slots")
    check = issue.get("check")
    if check not in {"shop_costs", "shop_names"} or not isinstance(slots, list):
        return False
    for slot_number in slots:
        slot = _runtime_shop_slot(recognition, slot_number)
        if not slot:
            return False
        if check == "shop_costs" and slot.get("cost_candidate", slot.get("cost")) is None:
            return False
        if check == "shop_names" and not (slot.get("name_candidate") or slot.get("name")):
            return False
    return True


def _runtime_shop_slot(recognition: dict[str, Any], slot_number: Any) -> dict[str, Any] | None:
    for slot in recognition.get("shop") or []:
        if isinstance(slot, dict) and str(slot.get("slot")) == str(slot_number):
            return slot
    return None


def _has_runtime_shop_evidence(recognition: dict[str, Any]) -> bool:
    occupied = [slot for slot in recognition.get("shop") or [] if isinstance(slot, dict) and slot.get("state") == "occupied"]
    return any((slot.get("name_candidate") or slot.get("name") or slot.get("cost_candidate") is not None or slot.get("cost") is not None) for slot in occupied)


def _runtime_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    layout_counts: dict[str, int] = {}
    readiness_counts: dict[str, int] = {}
    main_blockers: dict[str, int] = {}
    for record in records:
        state = record.get("state") if isinstance(record.get("state"), dict) else {}
        layout = str(state.get("layout") or "unknown")
        readiness = str(state.get("readiness") or "unknown")
        layout_counts[layout] = layout_counts.get(layout, 0) + 1
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        blockers = state.get("blockers") if isinstance(state, dict) else []
        if isinstance(blockers, list) and blockers:
            code = str(blockers[0].get("code") or "unknown_blocker") if isinstance(blockers[0], dict) else "unknown_blocker"
            main_blockers[code] = main_blockers.get(code, 0) + 1
    return {
        "total_frames": len(records),
        "layout_counts": layout_counts,
        "readiness_counts": readiness_counts,
        "normal_shop_ready_count": _count(records, "normal_shop", "ready"),
        "augment_ready_count": _count(records, "augment", "ready"),
        "combat_ready_count": _count(records, "combat", "ready"),
        "contaminated_count": readiness_counts.get("contaminated", 0),
        "unknown_count": layout_counts.get("unknown", 0),
        "main_blockers": main_blockers,
    }


def _count(records: list[dict[str, Any]], layout: str, readiness: str) -> int:
    count = 0
    for record in records:
        state = record.get("state") if isinstance(record.get("state"), dict) else {}
        if state.get("layout") == layout and state.get("readiness") == readiness:
            count += 1
    return count


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_contact_sheet(path: Path, records: list[dict[str, Any]]) -> None:
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for record in records:
        image_path = Path(str(record.get("image_path") or ""))
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((320, 180))
            canvas = Image.new("RGB", (320, 216), color=(24, 24, 24))
            canvas.paste(thumb, ((320 - thumb.width) // 2, 0))
            thumbs.append(canvas)
        state = record.get("state") if isinstance(record.get("state"), dict) else {}
        labels.append(f"#{record.get('ordinal')} {state.get('layout')} {state.get('readiness')}")
    if not thumbs:
        return
    columns = min(4, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 216), color=(12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 320
        y = (index // columns) * 216
        sheet.paste(thumb, (x, y))
        draw.text((x + 8, y + 186), labels[index], fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
