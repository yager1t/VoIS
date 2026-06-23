# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Voice-to-Cursor Windows installer.

Builds two onedir executables from a shared Analysis/PYZ/COLLECT:
- voice-to-cursor.exe        : windowed (default GUI launcher)
- voice-to-cursor-console.exe: console (dry-run / debug launcher)
"""

from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).parent
ENTRYPOINT = PROJECT_ROOT / "installer" / "entrypoint.py"

# Data files to ship next to the executable.
datas = [
    (str(PROJECT_ROOT / ".env.template"), "."),
    (str(PROJECT_ROOT / "data" / "vocab"), "data/vocab"),
]

# Modules PyInstaller may fail to detect automatically.
hiddenimports = [
    "src",
    "src.main",
    "src.app",
    "src.config",
    "src.logging_config",
    "src.ui.settings_window",
    "src.ui.tray",
    "src.ui.vocab_editor",
    "src.asr.whisper_provider",
    "src.asr.final_transcriber",
    "src.asr.streaming",
    "src.audio.capture",
    "src.audio.vad",
    "src.hotkey.windows",
    "src.injection.windows",
    "src.postprocess.llm_client",
    "faster_whisper",
    "faster_whisper.transcribe",
    "faster_whisper.tokenizer",
    "faster_whisper.utils",
    "faster_whisper.feature_extractor",
    "torch",
    "torch._C",
    "torch._tensor",
    "torch.jit",
    "torch.nn",
    "torch.nn.functional",
    "torchaudio",
    "numpy",
    "numpy.core._dtype_ctypes",
    "sounddevice",
    "soundfile",
    "webrtcvad",
    "pynput",
    "pynput._util.win32",
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "pyperclip",
    "httpx",
    "loguru",
    "yaml",
    "pydantic",
    "pydantic_settings",
]

# Avoid bundling GPU-only torch artifacts if they exist.
excludes = [
    "torchvision",
    "torchaudio",
    "torch.testing._internal",
]

a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PROJECT_ROOT / "installer" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe_windowed = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="voice-to-cursor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    hide_console="hide-early",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

exe_console = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="voice-to-cursor-console",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_windowed,
    exe_console,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="voice-to-cursor",
)
