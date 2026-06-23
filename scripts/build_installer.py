"""Build the Voice-to-Cursor Windows executable and installer.

Usage:
    .venv\\Scripts\\python.exe scripts\\build_installer.py [--zip]

The script runs PyInstaller from the project root, then compiles an Inno Setup
installer if iscc.exe is available. A portable zip archive is also produced
unless --no-zip is passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
PYINSTALLER_DIST = DIST_DIR / "voice-to-cursor"
INSTALLER_DIR = DIST_DIR / "installer"
SPEC_FILE = PROJECT_ROOT / "installer" / "voice-to-cursor.spec"
ISS_FILE = PROJECT_ROOT / "installer" / "voice-to-cursor.iss"
ISCC = Path(r"C:\Program Files (x86)\Inno Setup 6\iscc.exe")

# Derive the version from the package so version strings live in one place.
sys.path.insert(0, str(PROJECT_ROOT))
import src

VERSION = src.__version__
ZIP_NAME = f"Voice-to-Cursor-{VERSION}-portable.zip"


def run(cmd: list[str | Path], *, cwd: Path | None = None) -> None:
    """Run a command and stream output."""
    print(f">>> {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def clean() -> None:
    """Remove previous PyInstaller and installer outputs."""
    for path in [PYINSTALLER_DIST, BUILD_DIR]:
        if path.exists():
            print(f"Removing {path}")
            shutil.rmtree(path)
    if INSTALLER_DIR.exists():
        print(f"Removing {INSTALLER_DIR}")
        shutil.rmtree(INSTALLER_DIR)
    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)


def build_executable() -> None:
    """Run PyInstaller with the project spec."""
    pyinstaller = PROJECT_ROOT / ".venv" / "Scripts" / "pyinstaller.exe"
    run(
        [
            str(pyinstaller),
            "--clean",
            "--noconfirm",
            str(SPEC_FILE),
        ],
        cwd=PROJECT_ROOT,
    )


def write_launch_bat() -> None:
    """Write a portable launcher for the zip distribution."""
    launcher = PYINSTALLER_DIST / "launch.bat"
    launcher.write_text(
        "@echo off\n"
        "cd /d \"%~dp0\"\n"
        "start \"\" \"voice-to-cursor.exe\"\n",
        encoding="utf-8",
    )
    console_launcher = PYINSTALLER_DIST / "launch-dry-run.bat"
    console_launcher.write_text(
        "@echo off\n"
        "cd /d \"%~dp0\"\n"
        "voice-to-cursor-console.exe --dry-run\n"
        "pause\n",
        encoding="utf-8",
    )


def build_zip() -> Path:
    """Create a zip archive of the PyInstaller onedir distribution."""
    zip_path = INSTALLER_DIR / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PYINSTALLER_DIST.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(PYINSTALLER_DIST)
                zf.write(file_path, arcname)
    return zip_path


def build_inno_installer() -> Path | None:
    """Compile the Inno Setup installer, if available."""
    if not ISCC.exists():
        print("Inno Setup compiler not found; skipping .exe installer.")
        return None
    run([str(ISCC), f"/DMyAppVersion={VERSION}", str(ISS_FILE)], cwd=PROJECT_ROOT)
    candidates = list(INSTALLER_DIR.glob("Voice-to-Cursor-*-Setup.exe"))
    return candidates[0] if candidates else None


def human_size(path: Path) -> str:
    """Return a human-readable file size."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main(argv: list[str] | None = None) -> int:
    """Entry point for the build script."""
    parser = argparse.ArgumentParser(description="Build Voice-to-Cursor installer")
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip creating the portable zip archive.",
    )
    parser.add_argument(
        "--no-inno",
        action="store_true",
        help="Skip compiling the Inno Setup installer.",
    )
    args = parser.parse_args(argv)

    start = time.monotonic()
    clean()

    print("\n=== Building PyInstaller executable ===")
    build_executable()
    write_launch_bat()

    artifacts: list[Path] = []

    if not args.no_zip:
        print("\n=== Building portable zip archive ===")
        artifacts.append(build_zip())

    if not args.no_inno:
        print("\n=== Building Inno Setup installer ===")
        installer = build_inno_installer()
        if installer:
            artifacts.append(installer)

    elapsed = time.monotonic() - start
    print(f"\n=== Build completed in {elapsed:.1f}s ===")
    for artifact in artifacts:
        print(f"  {artifact.relative_to(PROJECT_ROOT)}  ({human_size(artifact)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
