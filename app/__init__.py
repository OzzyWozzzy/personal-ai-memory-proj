"""Temporal memory graph application package.

Importing an application entry point loads gitignored local configuration with
deterministic precedence. Existing process variables remain authoritative.
"""

from app.environment import load_local_environment as _load_local_environment

_load_local_environment()

del _load_local_environment
