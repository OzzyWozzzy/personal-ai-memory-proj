"""Deterministic local environment loading for repository entry points.

Process configuration remains authoritative. For local development the app reads
gitignored ``.env.local`` and ``.env`` files.  Values that already exist in the
process are never replaced, and the more-specific ``.env.local`` file wins over
``.env``.

The loader deliberately resolves files relative to the repository rather than
the current working directory.  It also avoids dotenv interpolation so the
result cannot vary with unrelated ambient shell variables.
"""
from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

from dotenv import dotenv_values


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE_NAMES = (".env.local", ".env")
_VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


def load_local_environment(
    *,
    root: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Load local dotenv files without replacing process values.

    Precedence, from highest to lowest, is:

    1. values already present in the process environment;
    2. ``.env.local``;
    3. ``.env``.

    Set ``MEMORY_LOAD_LOCAL_ENV=0`` (or ``false``, ``no``, or ``off``) in the
    process environment before importing :mod:`app` to disable local file
    loading.  The returned paths are the files that were found and parsed;
    values are intentionally never logged.
    """
    target = os.environ if environ is None else environ
    enabled = target.get("MEMORY_LOAD_LOCAL_ENV", "1").strip().lower()
    if enabled in _DISABLED_VALUES:
        return ()

    base = _REPO_ROOT if root is None else Path(root)
    process_keys = frozenset(target)
    file_keys: set[str] = set()
    loaded_files: list[Path] = []

    for file_name in _ENV_FILE_NAMES:
        path = base / file_name
        if not path.is_file():
            continue
        values = dotenv_values(path, interpolate=False)
        loaded_files.append(path)
        for key, value in values.items():
            if (
                value is None
                or not _VALID_ENV_NAME.fullmatch(key)
                or key in process_keys
                or key in file_keys
            ):
                continue
            target[key] = value
            file_keys.add(key)

    return tuple(loaded_files)
