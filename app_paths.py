"""Resolve bundled resources and writable desktop application data."""

import os
import sys
from pathlib import Path


def is_packaged():
    return bool(getattr(sys, "frozen", False))


def resource_root():
    return Path(__file__).resolve().parent


def install_root():
    return Path(sys.executable).resolve().parent if is_packaged() else resource_root()


def writable_root():
    if is_packaged():
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        return local / "PuYu"
    return resource_root()
