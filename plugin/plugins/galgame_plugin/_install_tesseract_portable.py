"""Install Tesseract without admin rights using /CURRENTUSER Inno Setup flag."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

INSTALLER_URL = (
    "https://ghproxy.com/https://github.com/UB-Mannheim/tesseract/"
    "releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
)
INSTALLER_NAME = "tesseract-ocr-w64-setup-5.4.0.20240606.exe"
TESSDATA_BASE_URL = "https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_fast@main"
LANGUAGES = ["chi_sim", "eng"]
TARGET_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\N.E.K.O\Tesseract-OCR"))


def download_file(url: str, destination: Path, timeout: float = 300.0) -> None:
    print(f"Downloading {url} ...")
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with destination.open("wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"  {pct:.1f}% ({downloaded}/{total} bytes)", end="\r")
    print()


def main() -> None:
    print("=" * 60)
    print("Tesseract Portable Installer (no admin required)")
    print("=" * 60)

    # Check if already installed
    exe_path = TARGET_DIR / "tesseract.exe"
    if exe_path.exists():
        print(f"Tesseract already exists at: {exe_path}")
        # Still ensure languages are present
    else:
        tmp_dir = Path(tempfile.gettempdir()) / "neko-tesseract-install"
        tmp_dir.mkdir(exist_ok=True)
        installer_path = tmp_dir / INSTALLER_NAME

        if not installer_path.exists() or installer_path.stat().st_size < 1_000_000:
            try:
                download_file(INSTALLER_URL, installer_path)
            except Exception as exc:
                print(f"Download failed: {exc}")
                sys.exit(1)
        else:
            print(f"Using cached installer: {installer_path}")

        print("\nRunning installer (no-admin mode)...")
        cmd = [
            str(installer_path),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/SP-",
            "/CURRENTUSER",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300)
            print("Installer completed.")
        except subprocess.CalledProcessError as exc:
            print(f"Installer failed with code {exc.returncode}")
            sys.exit(1)
        except Exception as exc:
            print(f"Installer failed: {exc}")
            sys.exit(1)

    # Ensure tessdata dir and languages
    tessdata_dir = TARGET_DIR / "tessdata"
    tessdata_dir.mkdir(parents=True, exist_ok=True)

    missing_langs = []
    for lang in LANGUAGES:
        data_file = tessdata_dir / f"{lang}.traineddata"
        if not data_file.exists():
            missing_langs.append(lang)

    if missing_langs:
        print(f"\nDownloading language files: {missing_langs}")
        for lang in missing_langs:
            url = f"{TESSDATA_BASE_URL}/{lang}.traineddata"
            dest = tessdata_dir / f"{lang}.traineddata"
            try:
                download_file(url, dest)
                print(f"  {lang}: OK")
            except Exception as exc:
                print(f"  {lang}: FAILED ({exc})")
                sys.exit(1)
    else:
        print("All required language files are present.")

    # Verify
    exe_path = TARGET_DIR / "tesseract.exe"
    if exe_path.exists():
        print(f"\nTesseract installed at: {exe_path}")
        print("Installation complete.")
    else:
        print(f"\nERROR: tesseract.exe not found at {exe_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
