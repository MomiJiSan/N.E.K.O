#!/usr/bin/env python3
"""Compare enrollment and runtime audio domains without retaining biometrics."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from enum import IntEnum
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SOURCE_SAMPLE_RATE_HZ = 48_000
TARGET_SAMPLE_RATE_HZ = 16_000
TARGET_AUDIO_SAMPLES = TARGET_SAMPLE_RATE_HZ * 3
HOLDOUT_FIRST_SAMPLES = TARGET_SAMPLE_RATE_HZ * 3 // 2
THRESHOLD = 0.40
RUNTIME_CHUNK_SAMPLES = 480
MINIMUM_SPEAKER_COUNT = 3
MAXIMUM_CASE_COUNT = 64
MINIMUM_SOURCE_SAMPLES = SOURCE_SAMPLE_RATE_HZ * 31 // 10
MAXIMUM_SOURCE_SECONDS = 4
MAXIMUM_MANIFEST_BYTES = 1_000_000
REPORT_SCHEMA_VERSION = 1
WORKLET_RUNNER = PROJECT_ROOT / "scripts" / "_voice_enrollment_worklet_runner.cjs"
WORKLET_SOURCE = PROJECT_ROOT / "static" / "audio-processor.js"
REQUIRED_SCENARIOS = frozenset({"quiet", "steady-noise", "natural-pause"})
PROTECTED_PROJECT_ROOTS = tuple(
    PROJECT_ROOT / name
    for name in (
        "app",
        "main_logic",
        "main_routers",
        "scripts",
        "static",
        "templates",
        "utils",
    )
)
ALLOWED_PROJECT_REPORT_ROOTS = frozenset({"reports", "artifacts", ".artifacts"})


class ExitCode(IntEnum):
    PASS = 0
    BLOCK = 2
    CORPUS_UNAVAILABLE = 3
    CAMPPLUS_UNAVAILABLE = 4
    RUNTIME_PREPROCESSOR_UNAVAILABLE = 5
    BROWSER_RESAMPLER_UNAVAILABLE = 6
    INTERNAL_ERROR = 70


class _HarnessFailure(RuntimeError):
    def __init__(self, exit_code: ExitCode, verdict: str) -> None:
        self.exit_code = exit_code
        self.verdict = verdict
        super().__init__(verdict)


@dataclass(frozen=True, slots=True)
class _CorpusCase:
    speaker_id: str
    reference_paths: tuple[Path, Path, Path]
    holdout_path: Path


def _empty_report(verdict: str, *, run_id: str) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "verdict": verdict,
        "speaker_count": 0,
        "case_count": 0,
        "device_class_count": 0,
        "scenario_count": 0,
        "decision_count": 0,
        "cross_threshold_disagreement_count": 0,
        "disagreements": {
            "reference_leave_one_out": 0,
            "holdout_1_5": 0,
            "holdout_3_0": 0,
        },
        "runtime_noise_reduction": "enabled",
    }


def _corpus_failure() -> _HarnessFailure:
    return _HarnessFailure(
        ExitCode.CORPUS_UNAVAILABLE,
        "CORPUS_UNAVAILABLE_OR_INVALID",
    )


def _safe_atom(value: object) -> str:
    if type(value) is not str:
        raise _corpus_failure()
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 128
        or "/" in normalized
        or "\\" in normalized
    ):
        raise _corpus_failure()
    return normalized


def _safe_audio_path(corpus_dir: Path, value: object) -> Path:
    if type(value) is not str or not value.strip():
        raise _corpus_failure()
    declared = Path(value)
    if declared.is_absolute():
        raise _corpus_failure()
    resolved = (corpus_dir / declared).resolve()
    try:
        resolved.relative_to(corpus_dir)
    except ValueError:
        raise _corpus_failure() from None
    if not resolved.is_file():
        raise _corpus_failure()
    return resolved


def _validate_source_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SOURCE_SAMPLE_RATE_HZ
                or source.getcomptype() != "NONE"
            ):
                raise _corpus_failure()
            frame_count = source.getnframes()
            if not (
                MINIMUM_SOURCE_SAMPLES
                <= frame_count
                <= SOURCE_SAMPLE_RATE_HZ * MAXIMUM_SOURCE_SECONDS
            ):
                raise _corpus_failure()
    except _HarnessFailure:
        raise
    except (OSError, EOFError, wave.Error):
        raise _corpus_failure() from None


def _load_manifest(
    corpus_dir: Path,
) -> tuple[list[_CorpusCase], int, int, int]:
    try:
        root = corpus_dir.resolve(strict=True)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size > MAXIMUM_MANIFEST_BYTES:
            raise _corpus_failure()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except _HarnessFailure:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _corpus_failure() from None

    if (
        type(payload) is not dict
        or payload.get("schema_version") != 1
        or payload.get("source_sample_rate_hz") != SOURCE_SAMPLE_RATE_HZ
    ):
        raise _corpus_failure()
    raw_cases = payload.get("cases")
    if type(raw_cases) is not list or not 1 <= len(raw_cases) <= MAXIMUM_CASE_COUNT:
        raise _corpus_failure()

    cases: list[_CorpusCase] = []
    speaker_ids: set[str] = set()
    device_classes: set[str] = set()
    scenarios: set[str] = set()
    all_paths: set[Path] = set()
    for raw_case in raw_cases:
        if type(raw_case) is not dict:
            raise _corpus_failure()
        speaker_id = _safe_atom(raw_case.get("speaker_id"))
        device_class = _safe_atom(raw_case.get("device_class"))
        scenario = _safe_atom(raw_case.get("scenario"))
        raw_references = raw_case.get("references")
        if type(raw_references) is not list or len(raw_references) != 3:
            raise _corpus_failure()
        references = tuple(
            _safe_audio_path(root, value) for value in raw_references
        )
        holdout = _safe_audio_path(root, raw_case.get("holdout"))
        owned_paths = (*references, holdout)
        if len(set(owned_paths)) != 4 or any(path in all_paths for path in owned_paths):
            raise _corpus_failure()
        for path in owned_paths:
            _validate_source_wav(path)
        all_paths.update(owned_paths)
        cases.append(
            _CorpusCase(
                speaker_id=speaker_id,
                reference_paths=(references[0], references[1], references[2]),
                holdout_path=holdout,
            )
        )
        speaker_ids.add(speaker_id)
        device_classes.add(device_class)
        scenarios.add(scenario)
    if len(speaker_ids) < MINIMUM_SPEAKER_COUNT:
        raise _corpus_failure()
    if not REQUIRED_SCENARIOS.issubset(scenarios):
        raise _corpus_failure()
    return cases, len(speaker_ids), len(device_classes), len(scenarios)


def _read_source_pcm16(path: Path) -> bytearray:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SOURCE_SAMPLE_RATE_HZ
                or source.getcomptype() != "NONE"
            ):
                raise _corpus_failure()
            frame_count = source.getnframes()
            if not (
                MINIMUM_SOURCE_SAMPLES
                <= frame_count
                <= SOURCE_SAMPLE_RATE_HZ * MAXIMUM_SOURCE_SECONDS
            ):
                raise _corpus_failure()
            pcm16 = bytearray(source.readframes(frame_count))
    except _HarnessFailure:
        raise
    except (OSError, EOFError, wave.Error):
        raise _corpus_failure() from None
    if len(pcm16) != frame_count * 2:
        _wipe_bytes(pcm16)
        raise _corpus_failure()
    return pcm16


def _wipe_bytes(value: bytearray | None) -> None:
    if value is not None:
        value[:] = b"\x00" * len(value)


def _wipe_array(value: np.ndarray | None) -> None:
    if value is None:
        return
    try:
        if not value.flags.writeable:
            value.setflags(write=True)
        value.fill(0)
    except Exception:
        pass


def _resolve_node(node: str) -> str:
    resolved = shutil.which(node)
    if resolved is None:
        raise _HarnessFailure(
            ExitCode.BROWSER_RESAMPLER_UNAVAILABLE,
            "BROWSER_RESAMPLER_UNAVAILABLE",
        )
    return resolved


def _run_browser_path(
    source_pcm16: bytearray,
    *,
    node: str,
) -> bytearray:
    try:
        completed = subprocess.run(
            [
                node,
                str(WORKLET_RUNNER),
                str(WORKLET_SOURCE),
                str(SOURCE_SAMPLE_RATE_HZ),
                str(TARGET_SAMPLE_RATE_HZ),
                str(TARGET_AUDIO_SAMPLES),
            ],
            input=bytes(source_pcm16),
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise _HarnessFailure(
            ExitCode.BROWSER_RESAMPLER_UNAVAILABLE,
            "BROWSER_RESAMPLER_UNAVAILABLE",
        ) from None
    if completed.returncode == int(ExitCode.CORPUS_UNAVAILABLE):
        raise _corpus_failure()
    if completed.returncode != 0:
        raise _HarnessFailure(
            ExitCode.BROWSER_RESAMPLER_UNAVAILABLE,
            "BROWSER_RESAMPLER_UNAVAILABLE",
        )
    result = bytearray(completed.stdout)
    if len(result) != TARGET_AUDIO_SAMPLES * 2:
        _wipe_bytes(result)
        raise _HarnessFailure(
            ExitCode.BROWSER_RESAMPLER_UNAVAILABLE,
            "BROWSER_RESAMPLER_UNAVAILABLE",
        )
    return result


async def _run_runtime_path(source_pcm16: bytearray) -> bytearray:
    try:
        from main_logic.voice_turn.audio_input import VoiceInputAudioPipeline
        from utils import audio_processor as audio_processor_module
    except Exception:
        raise _HarnessFailure(
            ExitCode.RUNTIME_PREPROCESSOR_UNAVAILABLE,
            "RUNTIME_PREPROCESSOR_UNAVAILABLE",
        ) from None

    # The CLI emits exactly one aggregate JSON report. Mute production audio
    # diagnostics only while this isolated evaluator owns the pipeline, then
    # restore the process-global logger state for import-based unit tests.
    audio_logger = audio_processor_module.logger
    logger_was_disabled = audio_logger.disabled
    audio_logger.disabled = True
    pipeline: Any | None = None
    output = bytearray()
    try:
        pipeline = VoiceInputAudioPipeline(nr_enabled=True)
        chunk_bytes = RUNTIME_CHUNK_SAMPLES * 2
        for offset in range(0, len(source_pcm16) - chunk_bytes + 1, chunk_bytes):
            chunk = bytes(memoryview(source_pcm16)[offset : offset + chunk_bytes])
            frame = await pipeline.process(
                chunk,
                sample_rate_hz=SOURCE_SAMPLE_RATE_HZ,
            )
            if not frame.rnnoise_available:
                raise _HarnessFailure(
                    ExitCode.RUNTIME_PREPROCESSOR_UNAVAILABLE,
                    "RUNTIME_PREPROCESSOR_UNAVAILABLE",
                )
            output.extend(frame.pcm16)
            if len(output) >= TARGET_AUDIO_SAMPLES * 2:
                del output[TARGET_AUDIO_SAMPLES * 2 :]
                return output
        raise _corpus_failure()
    except _HarnessFailure:
        _wipe_bytes(output)
        raise
    except Exception:
        _wipe_bytes(output)
        raise _HarnessFailure(
            ExitCode.RUNTIME_PREPROCESSOR_UNAVAILABLE,
            "RUNTIME_PREPROCESSOR_UNAVAILABLE",
        ) from None
    finally:
        unwinding = sys.exc_info()[0] is not None
        close_failure: BaseException | None = None
        try:
            if pipeline is not None:
                await pipeline.close()
        except BaseException as exc:
            close_failure = exc
            _wipe_bytes(output)
        finally:
            audio_logger.disabled = logger_was_disabled
        if close_failure is not None and not unwinding:
            if isinstance(close_failure, asyncio.CancelledError):
                raise close_failure
            raise _HarnessFailure(
                ExitCode.RUNTIME_PREPROCESSOR_UNAVAILABLE,
                "RUNTIME_PREPROCESSOR_UNAVAILABLE",
            ) from None


def _create_model(asset_dir: Path | None) -> Any:
    model: Any | None = None
    try:
        from main_logic.asr_client.speaker_shadow.campplus import (
            CampPlusEmbeddingModel,
        )
        model = CampPlusEmbeddingModel(asset_dir=asset_dir)
        if model.load():
            return model
    except Exception:
        pass
    if model is not None:
        try:
            model.close()
        except Exception:
            pass
    raise _HarnessFailure(
        ExitCode.CAMPPLUS_UNAVAILABLE,
        "CAMPPLUS_MODEL_UNAVAILABLE",
    )


def _normalized_sum(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("embedding set must not be empty")
    result = np.zeros(embeddings[0].shape, dtype=np.float32)
    try:
        for embedding in embeddings:
            if embedding.shape != result.shape or not np.isfinite(embedding).all():
                raise ValueError("embedding contract mismatch")
            np.add(result, embedding, out=result)
        norm = float(np.linalg.norm(result))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("embedding norm is invalid")
        np.divide(result, np.float32(norm), out=result)
        return result
    except BaseException:
        _wipe_array(result)
        raise


def _embedding(
    model: Any,
    pcm16: bytearray,
    *,
    sample_count: int,
) -> np.ndarray:
    bounded = bytes(memoryview(pcm16)[: sample_count * 2])
    try:
        return model.embedding_from_pcm16(
            bounded,
            sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        )
    except Exception:
        raise _HarnessFailure(
            ExitCode.CAMPPLUS_UNAVAILABLE,
            "CAMPPLUS_MODEL_UNAVAILABLE",
        ) from None


def _path_decisions(
    model: Any,
    references_pcm16: Sequence[bytearray],
    holdout_pcm16: bytearray,
) -> tuple[bool, bool, bool, bool, bool]:
    embeddings: list[np.ndarray] = []
    leave_one_out: list[np.ndarray] = []
    centroid: np.ndarray | None = None
    holdout_first: np.ndarray | None = None
    holdout_full: np.ndarray | None = None
    try:
        embeddings = [
            _embedding(model, pcm16, sample_count=TARGET_AUDIO_SAMPLES)
            for pcm16 in references_pcm16
        ]
        leave_one_out = [
            _normalized_sum((embeddings[1], embeddings[2])),
            _normalized_sum((embeddings[0], embeddings[2])),
            _normalized_sum((embeddings[0], embeddings[1])),
        ]
        reference_decisions = tuple(
            float(np.dot(embedding, other_reference)) >= THRESHOLD
            for embedding, other_reference in zip(
                embeddings,
                leave_one_out,
                strict=True,
            )
        )
        centroid = _normalized_sum(embeddings)
        holdout_first = _embedding(
            model,
            holdout_pcm16,
            sample_count=HOLDOUT_FIRST_SAMPLES,
        )
        holdout_full = _embedding(
            model,
            holdout_pcm16,
            sample_count=TARGET_AUDIO_SAMPLES,
        )
        return (
            bool(reference_decisions[0]),
            bool(reference_decisions[1]),
            bool(reference_decisions[2]),
            float(np.dot(centroid, holdout_first)) >= THRESHOLD,
            float(np.dot(centroid, holdout_full)) >= THRESHOLD,
        )
    finally:
        for value in embeddings:
            _wipe_array(value)
        for value in leave_one_out:
            _wipe_array(value)
        _wipe_array(centroid)
        _wipe_array(holdout_first)
        _wipe_array(holdout_full)


async def _prepare_case_audio(
    case: _CorpusCase,
    *,
    node: str,
) -> tuple[list[bytearray], bytearray, list[bytearray], bytearray]:
    enrollment: list[bytearray] = []
    runtime: list[bytearray] = []
    try:
        for path in (*case.reference_paths, case.holdout_path):
            source: bytearray | None = None
            enrollment_pcm: bytearray | None = None
            runtime_pcm: bytearray | None = None
            try:
                source = _read_source_pcm16(path)
                enrollment_pcm = _run_browser_path(source, node=node)
                runtime_pcm = await _run_runtime_path(source)
                enrollment.append(enrollment_pcm)
                runtime.append(runtime_pcm)
                enrollment_pcm = None
                runtime_pcm = None
            finally:
                _wipe_bytes(source)
                _wipe_bytes(enrollment_pcm)
                _wipe_bytes(runtime_pcm)
        return enrollment[:3], enrollment[3], runtime[:3], runtime[3]
    except BaseException:
        for value in enrollment:
            _wipe_bytes(value)
        for value in runtime:
            _wipe_bytes(value)
        raise


async def _evaluate(
    cases: Sequence[_CorpusCase],
    *,
    speaker_count: int,
    device_class_count: int = 0,
    scenario_count: int = 0,
    asset_dir: Path | None,
    node: str,
    run_id: str | None = None,
) -> tuple[ExitCode, dict[str, object]]:
    effective_run_id = run_id or uuid.uuid4().hex
    model = _create_model(asset_dir)
    reference_disagreements = 0
    holdout_first_disagreements = 0
    holdout_full_disagreements = 0
    try:
        for case in cases:
            enrollment_refs: list[bytearray] = []
            runtime_refs: list[bytearray] = []
            enrollment_holdout: bytearray | None = None
            runtime_holdout: bytearray | None = None
            try:
                (
                    enrollment_refs,
                    enrollment_holdout,
                    runtime_refs,
                    runtime_holdout,
                ) = await _prepare_case_audio(case, node=node)
                enrollment_decisions = _path_decisions(
                    model,
                    enrollment_refs,
                    enrollment_holdout,
                )
                runtime_decisions = _path_decisions(
                    model,
                    runtime_refs,
                    runtime_holdout,
                )
                reference_disagreements += sum(
                    left != right
                    for left, right in zip(
                        enrollment_decisions[:3],
                        runtime_decisions[:3],
                        strict=True,
                    )
                )
                holdout_first_disagreements += int(
                    enrollment_decisions[3] != runtime_decisions[3]
                )
                holdout_full_disagreements += int(
                    enrollment_decisions[4] != runtime_decisions[4]
                )
            finally:
                for value in enrollment_refs:
                    _wipe_bytes(value)
                for value in runtime_refs:
                    _wipe_bytes(value)
                _wipe_bytes(enrollment_holdout)
                _wipe_bytes(runtime_holdout)
    finally:
        model.close()

    total_disagreements = (
        reference_disagreements
        + holdout_first_disagreements
        + holdout_full_disagreements
    )
    verdict = (
        "PASS_KEEP_16K_CONTRACT"
        if total_disagreements == 0
        else "BLOCK_AUDIO_NORMALIZATION_REQUIRED"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": effective_run_id,
        "verdict": verdict,
        "speaker_count": speaker_count,
        "case_count": len(cases),
        "device_class_count": device_class_count,
        "scenario_count": scenario_count,
        "decision_count": len(cases) * 5,
        "cross_threshold_disagreement_count": total_disagreements,
        "disagreements": {
            "reference_leave_one_out": reference_disagreements,
            "holdout_1_5": holdout_first_disagreements,
            "holdout_3_0": holdout_full_disagreements,
        },
        "runtime_noise_reduction": "enabled",
    }
    return (
        ExitCode.PASS if total_disagreements == 0 else ExitCode.BLOCK,
        report,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--campplus-asset-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--node", default="node")
    return parser


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_has_symlink_component(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            return True
    return False


def _prepare_output_path(
    output: Path | None,
    *,
    corpus_dir: Path,
    asset_dir: Path | None,
) -> Path | None:
    if output is None:
        return None
    absolute = output.expanduser().absolute()
    if absolute.suffix.lower() != ".json" or _path_has_symlink_component(absolute):
        raise _HarnessFailure(
            ExitCode.INTERNAL_ERROR,
            "HARNESS_INTERNAL_ERROR",
        )
    resolved = absolute.resolve(strict=False)
    protected_roots = [
        root.resolve(strict=False) for root in PROTECTED_PROJECT_ROOTS
    ]
    protected_roots.append(corpus_dir.resolve(strict=False))
    if asset_dir is not None:
        protected_roots.append(asset_dir.resolve(strict=False))
    if any(_path_is_within(resolved, root) for root in protected_roots):
        raise _HarnessFailure(
            ExitCode.INTERNAL_ERROR,
            "HARNESS_INTERNAL_ERROR",
        )
    try:
        project_relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        project_relative = None
    if project_relative is not None:
        first_part = project_relative.parts[0] if project_relative.parts else ""
        if (
            first_part not in ALLOWED_PROJECT_REPORT_ROOTS
            and not first_part.startswith(".pytest-")
        ):
            raise _HarnessFailure(
                ExitCode.INTERNAL_ERROR,
                "HARNESS_INTERNAL_ERROR",
            )
    if absolute.exists():
        raise _HarnessFailure(
            ExitCode.INTERNAL_ERROR,
            "HARNESS_INTERNAL_ERROR",
        )
    return absolute


def _write_report_atomic(output: Path, rendered: str) -> None:
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if _path_has_symlink_component(output) or output.exists():
            raise OSError("unsafe output target")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            stream.write(rendered)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _run(argv: Sequence[str] | None = None) -> tuple[int, dict[str, object], Path | None]:
    args = _build_parser().parse_args(argv)
    run_id = uuid.uuid4().hex
    output: Path | None = None
    try:
        output = _prepare_output_path(
            args.output,
            corpus_dir=args.corpus_dir,
            asset_dir=args.campplus_asset_dir,
        )
        cases, speaker_count, device_class_count, scenario_count = _load_manifest(
            args.corpus_dir
        )
        node = _resolve_node(args.node)
        asset_dir = (
            args.campplus_asset_dir.resolve()
            if args.campplus_asset_dir is not None
            else None
        )
        exit_code, report = asyncio.run(
            _evaluate(
                cases,
                speaker_count=speaker_count,
                device_class_count=device_class_count,
                scenario_count=scenario_count,
                asset_dir=asset_dir,
                node=node,
                run_id=run_id,
            )
        )
        return int(exit_code), report, output
    except _HarnessFailure as exc:
        return (
            int(exc.exit_code),
            _empty_report(exc.verdict, run_id=run_id),
            output,
        )
    except Exception:
        return (
            int(ExitCode.INTERNAL_ERROR),
            _empty_report("HARNESS_INTERNAL_ERROR", run_id=run_id),
            output,
        )


def main(argv: Sequence[str] | None = None) -> int:
    exit_code, report, output = _run(argv)
    rendered = json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if output is not None:
        try:
            _write_report_atomic(output, rendered)
        except OSError:
            exit_code = int(ExitCode.INTERNAL_ERROR)
            report = _empty_report(
                "HARNESS_INTERNAL_ERROR",
                run_id=str(report["run_id"]),
            )
            rendered = json.dumps(
                report,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
    print(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
