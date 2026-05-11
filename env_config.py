"""Shared .env loader and typed getters.

Lookup precedence: ``os.environ`` > parsed ``.env`` > caller default.
Keys are matched case-insensitively so a ``.env`` with legacy lowercase keys
(``sftp_user``) still resolves the canonical ``SFTP_USER``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_cache: dict[str, str] = {}
_loaded_from: Path | None = None


def load_env(env_file: Path | None) -> dict[str, str]:
    global _cache, _loaded_from
    _cache = {}
    _loaded_from = None
    if env_file is None or not env_file.exists():
        return _cache
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            _cache[key.upper()] = value
    _loaded_from = env_file
    return _cache


def _lookup(key: str) -> str | None:
    upper = key.upper()
    if upper in os.environ:
        return os.environ[upper]
    return _cache.get(upper)


def get_str(key: str, default: str | None = None, required: bool = False) -> str | None:
    value = _lookup(key)
    if value is None or value == "":
        if required:
            _missing(key)
        return default
    return value


def get_path(key: str, default: str | None = None, required: bool = False) -> Path | None:
    value = get_str(key, default=default, required=required)
    if value is None:
        return None
    return Path(value).expanduser()


def get_int(key: str, default: int) -> int:
    value = _lookup(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"warning: env {key}={value!r} is not an int; using default {default}", file=sys.stderr)
        return default


def _missing(key: str) -> None:
    hint = f" (loaded from {_loaded_from})" if _loaded_from else " (no .env loaded)"
    raise SystemExit(f"error: required env var {key} is not set{hint}")
