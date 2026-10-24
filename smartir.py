#!/usr/bin/env python3
"""One-command SMARTIR entry point for Raspberry Pi OS.

Run ``python smartir.py`` on the Pi desktop. On the first run this launcher
offers to install missing Raspberry Pi OS packages with apt. Camera and
libcamera packages cannot be installed solely inside this Git checkout.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "smartir_v2.py"
DEPENDENCIES = ROOT / "dependencies.txt"
CHECK_IMPORTS = "import cv2, numpy, picamera2, tkinter, evdev"


def on_raspberry_pi() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        return "raspberry pi" in Path("/proc/device-tree/model").read_bytes().decode(
            "utf-8", "ignore"
        ).lower()
    except OSError:
        return False


def os_python() -> str:
    candidate = Path("/usr/bin/python3")
    return str(candidate) if platform.system() == "Linux" and candidate.exists() else sys.executable


def apt_packages() -> list[str]:
    try:
        lines = DEPENDENCIES.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Cannot read {DEPENDENCIES}: {exc}") from exc
    packages = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not packages or any(not all(c.isascii() and (c.isalnum() or c in "-+.") for c in package)
                           for package in packages):
        raise RuntimeError("dependencies.txt must contain one apt package name per line")
    return packages


def missing_imports(python: str) -> str | None:
    result = subprocess.run(
        [python, "-c", CHECK_IMPORTS],
        capture_output=True, text=True, check=False,
    )
    return None if result.returncode == 0 else (result.stderr.strip() or "Import check failed")


def ensure_dependencies(python: str) -> None:
    if not on_raspberry_pi():
        return
    problem = missing_imports(python)
    if problem is None:
        return
    packages = apt_packages()
    print("SMARTIR needs Raspberry Pi OS camera/input packages:", file=sys.stderr)
    print("  " + " ".join(packages), file=sys.stderr)
    print(f"Import check: {problem}", file=sys.stderr)
    print("These packages install system-wide through apt; libcamera cannot be vendored in this folder.",
          file=sys.stderr)
    if not sys.stdin.isatty():
        raise RuntimeError("Run interactively once, or install packages with: "
                           "sudo apt update && xargs -a dependencies.txt sudo apt install -y")
    answer = input("Install the packages with sudo apt now? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("Installation declined. Install the packages and run again.")
    subprocess.run(["sudo", "apt-get", "update"], check=True)
    subprocess.run(["sudo", "apt-get", "install", "-y", *packages], check=True)
    problem = missing_imports(python)
    if problem is not None:
        raise RuntimeError(
            "Packages installed, but the OS Python still cannot import them. "
            "Do not create a manual library symlink; inspect package health.\n"
            f"Import error: {problem}"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not CORE.is_file():
        print(f"SMARTIR error: missing {CORE}", file=sys.stderr)
        return 1
    if arguments and arguments[0] in ("-h", "--help"):
        print("Usage: python smartir.py [doctor|snapshot|probe|calibrate|run|gui] [options]")
        print("With no command: open the GUI on a Pi desktop, or run diagnostics on Lite.")
        print("For full options after setup: python smartir.py gui --help")
        return 0
    python = os_python()
    try:
        ensure_dependencies(python)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"SMARTIR setup error: {exc}", file=sys.stderr)
        return 1
    if not arguments:
        arguments = ["gui"] if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else ["doctor", "--camera-test"]
    if arguments[0] == "gui" and platform.system() == "Linux" and os.geteuid() == 0:
        print("SMARTIR error: launch the GUI as your signed-in desktop user, not root. "
              "Type 'exit' to leave the root shell, then run 'python3 smartir.py'.",
              file=sys.stderr)
        return 1
    try:
        return subprocess.run([python, str(CORE), *arguments], check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
