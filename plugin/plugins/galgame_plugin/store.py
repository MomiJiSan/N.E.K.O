from __future__ import annotations

from typing import Any

from .models import (
    MODES,
    STORE_BOUND_GAME_ID,
    STORE_DEDUPE_WINDOW,
    STORE_EVENTS_BYTE_OFFSET,
    STORE_EVENTS_FILE_SIZE,
    STORE_LAST_ERROR,
    STORE_LAST_SEQ,
    STORE_MODE,
    STORE_OCR_CAPTURE_PROFILES,
    STORE_OCR_WINDOW_TARGET,
    STORE_PUSH_NOTIFICATIONS,
    STORE_SESSION_ID,
)


class GalgameStore:
    def __init__(self, plugin_store, logger) -> None:
        self._store = plugin_store
        self._logger = logger

    def _read(self, key: str, default: Any) -> Any:
        if not getattr(self._store, "enabled", False):
            return default
        return self._store._read_value(key, default)  # type: ignore[attr-defined]

    def _write(self, key: str, value: Any) -> None:
        if not getattr(self._store, "enabled", False):
            return
        self._store._write_value(key, value)  # type: ignore[attr-defined]

    @staticmethod
    def _sanitize_ocr_capture_profiles(raw_value: Any) -> tuple[dict[str, dict[str, float]], list[str]]:
        warnings: list[str] = []
        if raw_value in ({}, None):
            return {}, warnings
        if not isinstance(raw_value, dict):
            return {}, ["invalid ocr_capture_profiles dropped: non-object"]

        normalized: dict[str, dict[str, float]] = {}
        ratio_keys = (
            "left_inset_ratio",
            "right_inset_ratio",
            "top_ratio",
            "bottom_inset_ratio",
        )
        for process_name, profile in raw_value.items():
            if not isinstance(process_name, str) or not process_name.strip():
                warnings.append("invalid ocr_capture_profiles item dropped: bad process name")
                continue
            if not isinstance(profile, dict):
                warnings.append(
                    f"invalid ocr_capture_profiles item dropped: {process_name!r} is not an object"
                )
                continue
            cleaned: dict[str, float] = {}
            valid = True
            for key in ratio_keys:
                value = profile.get(key)
                if isinstance(value, bool):
                    valid = False
                    break
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    valid = False
                    break
                if parsed < 0.0 or parsed >= 1.0:
                    valid = False
                    break
                cleaned[key] = parsed
            if not valid:
                warnings.append(
                    f"invalid ocr_capture_profiles item dropped: {process_name!r} has invalid ratios"
                )
                continue
            normalized[process_name.strip()] = cleaned
        return normalized, warnings

    @staticmethod
    def _sanitize_ocr_window_target(raw_value: Any) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        if raw_value in ({}, None):
            return {}, warnings
        if not isinstance(raw_value, dict):
            return {}, ["invalid ocr_window_target dropped: non-object"]

        mode = str(raw_value.get("mode") or "auto").strip().lower()
        if mode not in {"auto", "manual"}:
            warnings.append("invalid ocr_window_target mode dropped: fallback to auto")
            mode = "auto"

        normalized_title = str(raw_value.get("normalized_title") or "").strip().lower()
        process_name = str(raw_value.get("process_name") or "").strip()
        window_key = str(raw_value.get("window_key") or "").strip()
        selected_at = str(raw_value.get("selected_at") or "").strip()

        try:
            pid = int(raw_value.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
            warnings.append("invalid ocr_window_target pid dropped: fallback to 0")
        try:
            last_known_hwnd = int(raw_value.get("last_known_hwnd") or 0)
        except (TypeError, ValueError):
            last_known_hwnd = 0
            warnings.append("invalid ocr_window_target hwnd dropped: fallback to 0")

        normalized = {
            "mode": mode,
            "window_key": window_key,
            "process_name": process_name,
            "normalized_title": normalized_title,
            "pid": max(0, pid),
            "last_known_hwnd": max(0, last_known_hwnd),
            "selected_at": selected_at,
        }

        if mode == "manual" and not any(
            [
                normalized["window_key"],
                normalized["process_name"],
                normalized["normalized_title"],
                normalized["pid"],
                normalized["last_known_hwnd"],
            ]
        ):
            warnings.append("invalid ocr_window_target dropped: empty manual target")
            return {}, warnings
        return normalized, warnings

    def load(self) -> tuple[dict[str, Any], list[str]]:
        warnings: list[str] = []
        raw_mode = self._read(STORE_MODE, "")
        mode = raw_mode if isinstance(raw_mode, str) and raw_mode in MODES else "companion"
        if raw_mode not in ("", mode):
            warnings.append(f"invalid store mode dropped: {raw_mode!r}")

        raw_window = self._read(STORE_DEDUPE_WINDOW, [])
        dedupe_window: list[dict[str, str]] = []
        if isinstance(raw_window, list):
            for item in raw_window:
                if not isinstance(item, dict):
                    warnings.append("invalid dedupe_window item dropped: non-object")
                    continue
                game_id = item.get("game_id")
                line_id = item.get("line_id")
                normalized_text = item.get("normalized_text")
                if not (
                    isinstance(game_id, str)
                    and isinstance(line_id, str)
                    and isinstance(normalized_text, str)
                ):
                    warnings.append("invalid dedupe_window item dropped: missing string fields")
                    continue
                dedupe_window.append(
                    {
                        "game_id": game_id,
                        "line_id": line_id,
                        "normalized_text": normalized_text,
                    }
                )
        elif raw_window not in (None, []):
            warnings.append("invalid dedupe_window dropped: non-array")

        raw_last_error = self._read(STORE_LAST_ERROR, {})
        last_error = dict(raw_last_error) if isinstance(raw_last_error, dict) else {}
        if raw_last_error not in ({}, last_error):
            warnings.append("invalid last_error dropped: non-object")
        ocr_capture_profiles, profile_warnings = self._sanitize_ocr_capture_profiles(
            self._read(STORE_OCR_CAPTURE_PROFILES, {})
        )
        warnings.extend(profile_warnings)
        ocr_window_target, target_warnings = self._sanitize_ocr_window_target(
            self._read(STORE_OCR_WINDOW_TARGET, {})
        )
        warnings.extend(target_warnings)

        restored = {
            STORE_BOUND_GAME_ID: self._read(STORE_BOUND_GAME_ID, ""),
            STORE_MODE: mode,
            STORE_PUSH_NOTIFICATIONS: bool(self._read(STORE_PUSH_NOTIFICATIONS, True)),
            STORE_SESSION_ID: self._read(STORE_SESSION_ID, ""),
            STORE_EVENTS_BYTE_OFFSET: max(0, int(self._read(STORE_EVENTS_BYTE_OFFSET, 0) or 0)),
            STORE_EVENTS_FILE_SIZE: max(0, int(self._read(STORE_EVENTS_FILE_SIZE, 0) or 0)),
            STORE_LAST_SEQ: max(0, int(self._read(STORE_LAST_SEQ, 0) or 0)),
            STORE_DEDUPE_WINDOW: dedupe_window,
            STORE_LAST_ERROR: last_error,
            STORE_OCR_CAPTURE_PROFILES: ocr_capture_profiles,
            STORE_OCR_WINDOW_TARGET: ocr_window_target,
        }
        if not isinstance(restored[STORE_BOUND_GAME_ID], str):
            warnings.append("invalid bound_game_id dropped: non-string")
            restored[STORE_BOUND_GAME_ID] = ""
        if not isinstance(restored[STORE_SESSION_ID], str):
            warnings.append("invalid session_id dropped: non-string")
            restored[STORE_SESSION_ID] = ""
        return restored, warnings

    def persist_preferences(
        self,
        *,
        bound_game_id: str,
        mode: str,
        push_notifications: bool,
    ) -> None:
        self._write(STORE_BOUND_GAME_ID, bound_game_id)
        self._write(STORE_MODE, mode)
        self._write(STORE_PUSH_NOTIFICATIONS, push_notifications)

    def persist_runtime(
        self,
        *,
        session_id: str,
        events_byte_offset: int,
        events_file_size: int,
        last_seq: int,
        dedupe_window: list[dict[str, str]],
        last_error: dict[str, Any],
    ) -> None:
        self._write(STORE_SESSION_ID, session_id)
        self._write(STORE_EVENTS_BYTE_OFFSET, max(0, int(events_byte_offset)))
        self._write(STORE_EVENTS_FILE_SIZE, max(0, int(events_file_size)))
        self._write(STORE_LAST_SEQ, max(0, int(last_seq)))
        self._write(STORE_DEDUPE_WINDOW, list(dedupe_window))
        self._write(STORE_LAST_ERROR, dict(last_error))

    def persist_ocr_capture_profiles(
        self,
        profiles: dict[str, dict[str, float]],
    ) -> None:
        self._write(
            STORE_OCR_CAPTURE_PROFILES,
            {
                str(process_name): {str(key): float(value) for key, value in profile.items()}
                for process_name, profile in profiles.items()
            },
        )

    def persist_ocr_window_target(self, target: dict[str, Any]) -> None:
        payload = dict(target or {})
        self._write(
            STORE_OCR_WINDOW_TARGET,
            {
                "mode": str(payload.get("mode") or "auto"),
                "window_key": str(payload.get("window_key") or ""),
                "process_name": str(payload.get("process_name") or ""),
                "normalized_title": str(payload.get("normalized_title") or ""),
                "pid": max(0, int(payload.get("pid") or 0)),
                "last_known_hwnd": max(0, int(payload.get("last_known_hwnd") or 0)),
                "selected_at": str(payload.get("selected_at") or ""),
            },
        )

    def clear_runtime(self) -> None:
        self.persist_runtime(
            session_id="",
            events_byte_offset=0,
            events_file_size=0,
            last_seq=0,
            dedupe_window=[],
            last_error={},
        )
