"""Setup wizard and launcher creation for Bark Infinity."""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_PORTS = {
    "webui": 7860,
    "streamlit": 8501,
}
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
ALLOWED_MODULES = {"torch", "transformers", "gradio", "streamlit"}


def get_required_modules(mode: str) -> list[str]:
    """Return required module names for the selected mode."""
    required = ["torch", "transformers"]
    if mode == "webui":
        required.append("gradio")
    elif mode == "streamlit":
        required.append("streamlit")
    return required


def find_missing_dependencies(modules: Iterable[str]) -> list[str]:
    """Return a list of missing modules."""
    missing = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def _validate_module_names(modules: Iterable[str]) -> None:
    invalid = [module for module in modules if not MODULE_NAME_PATTERN.fullmatch(module)]
    if invalid:
        raise ValueError(f"Invalid module name(s): {', '.join(invalid)}")
    unexpected = [module for module in modules if module not in ALLOWED_MODULES]
    if unexpected:
        raise ValueError(f"Unsupported module name(s): {', '.join(unexpected)}")


def install_missing_dependencies(modules: Iterable[str]) -> bool:
    """Install missing modules via pip."""
    modules = list(modules)
    if not modules:
        return True
    _validate_module_names(modules)
    print(f"Installing missing dependencies: {', '.join(modules)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--", *modules],
        check=False,
        shell=False,
        timeout=600,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Failed to install dependencies via pip.")
        if result.stderr:
            print(result.stderr.strip())
        elif result.stdout:
            print(result.stdout.strip())
    return result.returncode == 0


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    desktop_dir = Path.home() / "Desktop"
    if desktop_dir.is_dir():
        return desktop_dir
    return Path.cwd()


def _sanitize_filename(name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9 _.-]", "_", name)
    safe_name = safe_name.replace(os.sep, "_")
    if os.altsep:
        safe_name = safe_name.replace(os.altsep, "_")
    return safe_name


def _quote_windows(args: list[str]) -> str:
    return subprocess.list2cmdline(args)


def _build_command(mode: str, port: int, share: bool, platform_name: str) -> str:
    args = [sys.executable, "-m", "bark_infinity.cli", mode, "--port", str(port)]
    if share and mode == "webui":
        args.append("--share")
    if platform_name == "Windows":
        return _quote_windows(args)
    return " ".join(shlex.quote(arg) for arg in args)


def _render_launcher_content(platform_name: str, display_name: str, command: str) -> str:
    if platform_name == "Windows":
        return f"@echo off\n{command}\n"
    if platform_name == "Darwin":
        return f"#!/bin/bash\n{command}\n"
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={display_name}\n"
        f"Exec={command}\n"
        "Terminal=true\n"
    )


def create_launcher_script(
    mode: str = "webui",
    port: int | None = None,
    share: bool = False,
    output_dir: str | Path | None = None,
    platform_name: str | None = None,
) -> Path:
    """Create a one-click launcher script for the selected mode."""
    mode = mode.lower()
    if mode not in DEFAULT_PORTS:
        raise ValueError(f"Unsupported mode: {mode}")
    port = DEFAULT_PORTS[mode] if port is None else port
    platform_name = platform.system() if platform_name is None else platform_name
    output_path = _resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    display_name = (
        "Bark Infinity Web UI" if mode == "webui" else "Bark Infinity Streamlit UI"
    )
    if platform_name == "Windows":
        extension = ".bat"
    elif platform_name == "Darwin":
        extension = ".command"
    else:
        extension = ".desktop"

    command = _build_command(mode, port, share, platform_name)
    content = _render_launcher_content(platform_name, display_name, command)
    safe_name = _sanitize_filename(display_name)
    launcher_path = output_path / f"{safe_name}{extension}"
    launcher_path.write_text(content, encoding="utf-8")

    if platform_name in {"Linux", "Darwin"}:
        launcher_path.chmod(
            launcher_path.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

    print(f"Created launcher at: {launcher_path}")
    return launcher_path


def run_setup_wizard(
    mode: str = "webui",
    port: int | None = None,
    share: bool = False,
    create_shortcut: bool = True,
    install_missing: bool = False,
) -> dict[str, object]:
    """Run setup wizard for Bark Infinity."""
    mode = mode.lower()
    if mode not in DEFAULT_PORTS:
        raise ValueError(f"Unsupported mode: {mode}")
    port = DEFAULT_PORTS[mode] if port is None else port

    required = get_required_modules(mode)
    missing = find_missing_dependencies(required)

    if missing:
        print("Missing dependencies detected:")
        for module in missing:
            print(f"  - {module}")
        if install_missing:
            if install_missing_dependencies(missing):
                missing = find_missing_dependencies(required)
                if missing:
                    print("Still missing dependencies after install:")
                    for module in missing:
                        print(f"  - {module}")
            else:
                print("Failed to install missing dependencies.")
        else:
            print("Run with --install-missing to install them.")
    else:
        print("All required dependencies appear to be installed.")

    launcher_path = None
    if create_shortcut:
        launcher_path = create_launcher_script(mode=mode, port=port, share=share)

    return {"missing": missing, "launcher_path": launcher_path}
