from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator

import httpx

from .memory_reader import is_windows_platform
from .tesseract_support import _compute_phase_progress, _emit_progress
from .install_tasks import update_install_task_state

RAPIDOCR_PACKAGE_NAME = "rapidocr_onnxruntime"
DEFAULT_RAPIDOCR_ENGINE_TYPE = "onnxruntime"
DEFAULT_RAPIDOCR_LANG_TYPE = "ch"
DEFAULT_RAPIDOCR_MODEL_TYPE = "mobile"
DEFAULT_RAPIDOCR_OCR_VERSION = "PP-OCRv5"
DEFAULT_RAPIDOCR_PIP_SPEC = "rapidocr_onnxruntime"
DEFAULT_ONNXRUNTIME_PIP_SPEC = "onnxruntime"
_INSTALL_STATE_NAME = "install_state.json"
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def _expand_candidate_path(raw_path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(raw_path)))


def default_rapidocr_install_target_raw() -> str:
    if is_windows_platform():
        return "%LOCALAPPDATA%/Programs/N.E.K.O/RapidOCR"
    return ""


def resolve_rapidocr_install_target(raw_target_dir: str) -> Path:
    normalized = str(raw_target_dir or "").strip() or default_rapidocr_install_target_raw()
    if not normalized:
        return Path()
    return _expand_candidate_path(normalized)


def resolve_rapidocr_runtime_dir(raw_target_dir: str) -> Path:
    target_dir = resolve_rapidocr_install_target(raw_target_dir)
    return target_dir / "runtime" if target_dir else Path()


def resolve_rapidocr_site_packages_dir(raw_target_dir: str) -> Path:
    runtime_dir = resolve_rapidocr_runtime_dir(raw_target_dir)
    return runtime_dir / "site-packages" if runtime_dir else Path()


def resolve_rapidocr_model_cache_dir(raw_target_dir: str) -> Path:
    target_dir = resolve_rapidocr_install_target(raw_target_dir)
    return target_dir / "models" if target_dir else Path()


def _rapidocr_install_state_path(raw_target_dir: str) -> Path:
    target_dir = resolve_rapidocr_install_target(raw_target_dir)
    return target_dir / _INSTALL_STATE_NAME if target_dir else Path()


def rapidocr_selected_model_name(
    *,
    ocr_version: str,
    lang_type: str,
    model_type: str,
) -> str:
    return "/".join(
        [
            str(ocr_version or DEFAULT_RAPIDOCR_OCR_VERSION).strip() or DEFAULT_RAPIDOCR_OCR_VERSION,
            str(lang_type or DEFAULT_RAPIDOCR_LANG_TYPE).strip() or DEFAULT_RAPIDOCR_LANG_TYPE,
            str(model_type or DEFAULT_RAPIDOCR_MODEL_TYPE).strip() or DEFAULT_RAPIDOCR_MODEL_TYPE,
        ]
    )


def _default_install_manifest(
    *,
    engine_type: str,
    lang_type: str,
    model_type: str,
    ocr_version: str,
) -> dict[str, Any]:
    return {
        "name": "RapidOCR ONNXRuntime",
        "packages": [
            {"name": RAPIDOCR_PACKAGE_NAME, "spec": DEFAULT_RAPIDOCR_PIP_SPEC},
            {"name": "onnxruntime", "spec": DEFAULT_ONNXRUNTIME_PIP_SPEC},
        ],
        "engine_type": engine_type,
        "lang_type": lang_type,
        "model_type": model_type,
        "ocr_version": ocr_version,
    }


async def _load_install_manifest(
    *,
    manifest_url: str,
    timeout_seconds: float,
    engine_type: str,
    lang_type: str,
    model_type: str,
    ocr_version: str,
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    if not str(manifest_url or "").strip():
        return _default_install_manifest(
            engine_type=engine_type,
            lang_type=lang_type,
            model_type=model_type,
            ocr_version=ocr_version,
        )
    response = await client.get(
        str(manifest_url).strip(),
        headers={
            "Accept": "application/json",
            "User-Agent": "N.E.K.O/galgame_plugin",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("rapidocr install manifest returned an invalid payload")
    return payload


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for name in list(sys.modules.keys()):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


@contextmanager
def _rapidocr_import_context(
    *,
    site_packages_dir: Path,
    model_cache_dir: Path,
) -> Iterator[None]:
    inserted = False
    old_model_dir = os.environ.get("RAPIDOCR_MODEL_DIR")
    old_model_home = os.environ.get("RAPIDOCR_MODEL_HOME")
    dll_handles: list[Any] = []
    if site_packages_dir:
        site_packages_dir.mkdir(parents=True, exist_ok=True)
        site_path = str(site_packages_dir)
        if site_path not in sys.path:
            sys.path.insert(0, site_path)
            inserted = True
        if hasattr(os, "add_dll_directory"):
            for candidate in (
                site_packages_dir,
                site_packages_dir / "onnxruntime",
                site_packages_dir / "onnxruntime" / "capi",
            ):
                if candidate.is_dir():
                    try:
                        dll_handles.append(os.add_dll_directory(str(candidate)))
                    except OSError:
                        continue
    if model_cache_dir:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAPIDOCR_MODEL_DIR"] = str(model_cache_dir)
        os.environ["RAPIDOCR_MODEL_HOME"] = str(model_cache_dir)
    try:
        yield
    finally:
        for handle in dll_handles:
            try:
                handle.close()
            except Exception:
                pass
        if old_model_dir is None:
            os.environ.pop("RAPIDOCR_MODEL_DIR", None)
        else:
            os.environ["RAPIDOCR_MODEL_DIR"] = old_model_dir
        if old_model_home is None:
            os.environ.pop("RAPIDOCR_MODEL_HOME", None)
        else:
            os.environ["RAPIDOCR_MODEL_HOME"] = old_model_home
        if inserted:
            try:
                sys.path.remove(str(site_packages_dir))
            except ValueError:
                pass


def _rapidocr_package_dir(raw_target_dir: str) -> Path:
    site_packages_dir = resolve_rapidocr_site_packages_dir(raw_target_dir)
    return site_packages_dir / RAPIDOCR_PACKAGE_NAME if site_packages_dir else Path()


def _build_runtime_constructor_kwargs(
    runtime_class: type[Any],
    *,
    engine_type: str,
    lang_type: str,
    model_type: str,
    ocr_version: str,
    model_cache_dir: Path,
) -> dict[str, Any]:
    try:
        parameters = inspect.signature(runtime_class).parameters
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, Any] = {}
    direct_values = {
        "engine_type": engine_type,
        "lang_type": lang_type,
        "model_type": model_type,
        "ocr_version": ocr_version,
        "det_model_type": model_type,
        "cls_model_type": model_type,
        "rec_model_type": model_type,
        "cache_dir": str(model_cache_dir),
        "model_dir": str(model_cache_dir),
        "models_dir": str(model_cache_dir),
        "model_root": str(model_cache_dir),
    }
    for key, value in direct_values.items():
        if key in parameters:
            kwargs[key] = value
    return kwargs


def load_rapidocr_runtime(
    *,
    install_target_dir_raw: str,
    engine_type: str,
    lang_type: str,
    model_type: str,
    ocr_version: str,
    force_reload: bool = False,
) -> tuple[Any, dict[str, str]]:
    site_packages_dir = resolve_rapidocr_site_packages_dir(install_target_dir_raw)
    model_cache_dir = resolve_rapidocr_model_cache_dir(install_target_dir_raw)
    if not site_packages_dir:
        raise RuntimeError("missing RapidOCR site-packages directory")
    with _rapidocr_import_context(
        site_packages_dir=site_packages_dir,
        model_cache_dir=model_cache_dir,
    ):
        if force_reload:
            _purge_modules((RAPIDOCR_PACKAGE_NAME,))
        importlib.invalidate_caches()
        module = importlib.import_module(RAPIDOCR_PACKAGE_NAME)
        runtime_class = getattr(module, "RapidOCR", None)
        if runtime_class is None:
            raise RuntimeError("RapidOCR runtime class not found")
        runtime = runtime_class(
            **_build_runtime_constructor_kwargs(
                runtime_class,
                engine_type=engine_type,
                lang_type=lang_type,
                model_type=model_type,
                ocr_version=ocr_version,
                model_cache_dir=model_cache_dir,
            )
        )
    metadata = {
        "detected_path": str(Path(getattr(module, "__file__", "")).resolve().parent),
        "model_cache_dir": str(model_cache_dir),
        "selected_model": rapidocr_selected_model_name(
            ocr_version=ocr_version,
            lang_type=lang_type,
            model_type=model_type,
        ),
    }
    return runtime, metadata


def _write_install_state(
    *,
    raw_target_dir: str,
    metadata: dict[str, Any],
) -> None:
    state_path = _rapidocr_install_state_path(raw_target_dir)
    if not state_path:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def inspect_rapidocr_installation(
    *,
    install_target_dir_raw: str,
    engine_type: str = DEFAULT_RAPIDOCR_ENGINE_TYPE,
    lang_type: str = DEFAULT_RAPIDOCR_LANG_TYPE,
    model_type: str = DEFAULT_RAPIDOCR_MODEL_TYPE,
    ocr_version: str = DEFAULT_RAPIDOCR_OCR_VERSION,
    platform_fn: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    checker = platform_fn or is_windows_platform
    supported = bool(checker())
    target_dir = resolve_rapidocr_install_target(install_target_dir_raw)
    runtime_dir = resolve_rapidocr_runtime_dir(install_target_dir_raw)
    site_packages_dir = resolve_rapidocr_site_packages_dir(install_target_dir_raw)
    model_cache_dir = resolve_rapidocr_model_cache_dir(install_target_dir_raw)
    package_dir = _rapidocr_package_dir(install_target_dir_raw)
    install_state_path = _rapidocr_install_state_path(install_target_dir_raw)
    selected_model = rapidocr_selected_model_name(
        ocr_version=ocr_version,
        lang_type=lang_type,
        model_type=model_type,
    )
    detail = "missing"
    detected_path = str(package_dir) if package_dir.exists() else ""
    install_state: dict[str, Any] = {}
    runtime_error = ""

    if supported and install_state_path.is_file():
        try:
            install_state_payload = json.loads(install_state_path.read_text(encoding="utf-8"))
            if isinstance(install_state_payload, dict):
                install_state = install_state_payload
        except (OSError, ValueError, TypeError):
            install_state = {}

    if not supported:
        detail = "unsupported_platform"
    elif not package_dir.exists():
        detail = "missing"
    elif not install_state_path.is_file() or not model_cache_dir.exists():
        detail = "missing_models"
    else:
        try:
            _runtime, runtime_meta = load_rapidocr_runtime(
                install_target_dir_raw=install_target_dir_raw,
                engine_type=engine_type,
                lang_type=lang_type,
                model_type=model_type,
                ocr_version=ocr_version,
                force_reload=False,
            )
            detected_path = str(runtime_meta.get("detected_path") or detected_path)
            detail = "installed"
        except Exception as exc:
            detail = "broken_runtime"
            runtime_error = str(exc)

    installed = detail == "installed"
    return {
        "install_supported": supported,
        "installed": installed,
        "can_install": supported and not installed,
        "detected_path": detected_path,
        "target_dir": str(target_dir) if target_dir else "",
        "runtime_dir": str(runtime_dir) if runtime_dir else "",
        "site_packages_dir": str(site_packages_dir) if site_packages_dir else "",
        "model_cache_dir": str(model_cache_dir) if model_cache_dir else "",
        "selected_model": str(install_state.get("selected_model") or selected_model),
        "engine_type": str(install_state.get("engine_type") or engine_type),
        "lang_type": str(install_state.get("lang_type") or lang_type),
        "model_type": str(install_state.get("model_type") or model_type),
        "ocr_version": str(install_state.get("ocr_version") or ocr_version),
        "detail": detail,
        "runtime_error": runtime_error,
    }


def _blank_test_image() -> Any:
    import numpy as np
    from PIL import Image

    return np.asarray(Image.new("RGB", (64, 32), "white"))


def _warmup_rapidocr(
    *,
    install_target_dir_raw: str,
    engine_type: str,
    lang_type: str,
    model_type: str,
    ocr_version: str,
) -> dict[str, str]:
    runtime, metadata = load_rapidocr_runtime(
        install_target_dir_raw=install_target_dir_raw,
        engine_type=engine_type,
        lang_type=lang_type,
        model_type=model_type,
        ocr_version=ocr_version,
        force_reload=True,
    )
    test_image = _blank_test_image()
    _ = runtime(test_image)
    return metadata


def _run_pip_install(
    *,
    site_packages_dir: Path,
    packages: list[str],
    timeout_seconds: float,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--target",
        str(site_packages_dir),
        *packages,
    ]
    subprocess.run(
        command,
        check=True,
        timeout=timeout_seconds,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


async def install_rapidocr(
    *,
    logger,
    install_target_dir_raw: str,
    manifest_url: str,
    timeout_seconds: float,
    engine_type: str = DEFAULT_RAPIDOCR_ENGINE_TYPE,
    lang_type: str = DEFAULT_RAPIDOCR_LANG_TYPE,
    model_type: str = DEFAULT_RAPIDOCR_MODEL_TYPE,
    ocr_version: str = DEFAULT_RAPIDOCR_OCR_VERSION,
    force: bool = False,
    platform_fn: Callable[[], bool] | None = None,
    client_factory: Callable[[], Awaitable[httpx.AsyncClient] | httpx.AsyncClient] | None = None,
    task_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    install_status = inspect_rapidocr_installation(
        install_target_dir_raw=install_target_dir_raw,
        engine_type=engine_type,
        lang_type=lang_type,
        model_type=model_type,
        ocr_version=ocr_version,
        platform_fn=platform_fn,
    )
    if not install_status["install_supported"]:
        raise RuntimeError("RapidOCR install is only supported on Windows")
    if install_status["installed"] and not force:
        result = {
            **install_status,
            "already_installed": True,
            "summary": f"RapidOCR installed: {install_status['detected_path']}",
            "release_name": "RapidOCR ONNXRuntime",
            "asset_name": RAPIDOCR_PACKAGE_NAME,
        }
        if task_id:
            update_install_task_state(
                task_id,
                kind="rapidocr",
                status="completed",
                phase="completed",
                message="RapidOCR is already installed",
                progress=1.0,
                target_dir=str(install_status.get("target_dir") or ""),
                detected_path=str(install_status.get("detected_path") or ""),
            )
        await _emit_progress(
            progress_callback,
            {
                "status": "completed",
                "phase": "completed",
                "message": "RapidOCR is already installed",
                "progress": 1.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "resume_from": 0,
                "target_dir": str(install_status.get("target_dir") or ""),
                "detected_path": str(install_status.get("detected_path") or ""),
                "release_name": "RapidOCR ONNXRuntime",
                "asset_name": RAPIDOCR_PACKAGE_NAME,
            },
        )
        return result

    target_dir = resolve_rapidocr_install_target(install_target_dir_raw)
    if not target_dir:
        raise RuntimeError("missing RapidOCR install target directory")
    runtime_dir = resolve_rapidocr_runtime_dir(install_target_dir_raw)
    site_packages_dir = resolve_rapidocr_site_packages_dir(install_target_dir_raw)
    model_cache_dir = resolve_rapidocr_model_cache_dir(install_target_dir_raw)

    if task_id:
        update_install_task_state(
            task_id,
            kind="rapidocr",
            status="running",
            phase="metadata",
            message="Fetching RapidOCR install metadata",
            progress=_compute_phase_progress("metadata"),
            target_dir=str(target_dir),
        )
    await _emit_progress(
        progress_callback,
        {
            "status": "running",
            "phase": "metadata",
            "message": "Fetching RapidOCR install metadata",
            "progress": _compute_phase_progress("metadata"),
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "resume_from": 0,
            "target_dir": str(target_dir),
            "detected_path": "",
            "release_name": "",
            "asset_name": "",
        },
    )

    owned_client = False
    client: httpx.AsyncClient | None = None
    if client_factory is None:
        client = httpx.AsyncClient(
            timeout=timeout_seconds,
            trust_env=True,
            follow_redirects=True,
        )
        owned_client = True
    else:
        maybe_client = client_factory()
        client = await maybe_client if hasattr(maybe_client, "__await__") else maybe_client

    try:
        manifest = await _load_install_manifest(
            manifest_url=manifest_url,
            timeout_seconds=timeout_seconds,
            engine_type=engine_type,
            lang_type=lang_type,
            model_type=model_type,
            ocr_version=ocr_version,
            client=client,
        )
        release_name = str(manifest.get("name") or "RapidOCR ONNXRuntime")
        packages_obj = manifest.get("packages")
        package_specs = [
            str(item.get("spec") or "").strip()
            for item in packages_obj
            if isinstance(item, dict) and str(item.get("spec") or "").strip()
        ] if isinstance(packages_obj, list) else []
        if not package_specs:
            package_specs = [DEFAULT_RAPIDOCR_PIP_SPEC, DEFAULT_ONNXRUNTIME_PIP_SPEC]
        asset_name = ", ".join(package_specs)

        runtime_dir.mkdir(parents=True, exist_ok=True)
        site_packages_dir.mkdir(parents=True, exist_ok=True)
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        installing_progress = {
            "status": "running",
            "phase": "installing",
            "message": "Installing RapidOCR runtime packages",
            "progress": _compute_phase_progress("installing"),
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "resume_from": 0,
            "target_dir": str(target_dir),
            "detected_path": "",
            "release_name": release_name,
            "asset_name": asset_name,
            "error": "",
        }
        if task_id:
            update_install_task_state(task_id, kind="rapidocr", **installing_progress)
        await _emit_progress(progress_callback, installing_progress)

        await asyncio.to_thread(
            _run_pip_install,
            site_packages_dir=site_packages_dir,
            packages=package_specs,
            timeout_seconds=timeout_seconds,
        )

        verifying_progress = {
            "status": "running",
            "phase": "verifying",
            "message": "Warming up RapidOCR runtime",
            "progress": _compute_phase_progress("verifying"),
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "resume_from": 0,
            "target_dir": str(target_dir),
            "detected_path": "",
            "release_name": release_name,
            "asset_name": asset_name,
            "error": "",
        }
        if task_id:
            update_install_task_state(task_id, kind="rapidocr", **verifying_progress)
        await _emit_progress(progress_callback, verifying_progress)

        runtime_meta = await asyncio.to_thread(
            _warmup_rapidocr,
            install_target_dir_raw=install_target_dir_raw,
            engine_type=engine_type,
            lang_type=lang_type,
            model_type=model_type,
            ocr_version=ocr_version,
        )
        _write_install_state(
            raw_target_dir=install_target_dir_raw,
            metadata={
                "engine_type": engine_type,
                "lang_type": lang_type,
                "model_type": model_type,
                "ocr_version": ocr_version,
                "selected_model": runtime_meta["selected_model"],
                "detected_path": runtime_meta["detected_path"],
                "model_cache_dir": runtime_meta["model_cache_dir"],
            },
        )

        result_status = inspect_rapidocr_installation(
            install_target_dir_raw=install_target_dir_raw,
            engine_type=engine_type,
            lang_type=lang_type,
            model_type=model_type,
            ocr_version=ocr_version,
            platform_fn=platform_fn,
        )
        if not result_status["installed"]:
            raise RuntimeError(
                "RapidOCR installation is incomplete: "
                + str(result_status.get("detail") or "unknown")
            )
        result = {
            **result_status,
            "already_installed": False,
            "summary": f"RapidOCR installed to {result_status['target_dir']}",
            "release_name": release_name,
            "asset_name": asset_name,
        }
        completed_progress = {
            "status": "completed",
            "phase": "completed",
            "message": "RapidOCR installation completed",
            "progress": 1.0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "resume_from": 0,
            "target_dir": str(result_status.get("target_dir") or target_dir),
            "detected_path": str(result_status.get("detected_path") or ""),
            "release_name": release_name,
            "asset_name": asset_name,
            "error": "",
        }
        if task_id:
            update_install_task_state(task_id, kind="rapidocr", **completed_progress)
        await _emit_progress(progress_callback, completed_progress)
        return result
    except Exception as exc:
        error_message = str(exc)
        if task_id:
            update_install_task_state(
                task_id,
                kind="rapidocr",
                status="failed",
                phase="failed",
                message=error_message,
                progress=_compute_phase_progress("failed"),
                target_dir=str(target_dir),
                error=error_message,
            )
        await _emit_progress(
            progress_callback,
            {
                "status": "failed",
                "phase": "failed",
                "message": error_message,
                "progress": _compute_phase_progress("failed"),
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "resume_from": 0,
                "target_dir": str(target_dir),
                "detected_path": "",
                "release_name": "",
                "asset_name": "",
                "error": error_message,
            },
        )
        raise
    finally:
        if owned_client and client is not None:
            await client.aclose()
        try:
            _purge_modules((RAPIDOCR_PACKAGE_NAME,))
        except Exception as exc:
            if logger is not None:
                logger.warning("RapidOCR module cleanup failed: {}", exc)
