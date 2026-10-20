"""Admin CLI + worker-contract spec for the LLM bridge queue.

This module does not launch workers. Any local structured-output LLM worker can
implement the contract documented in ``WORKER_PROTOCOL`` below. Here we
provide only the operator tooling:

    python -m app.engine.bridge_admin status
    python -m app.engine.bridge_admin requeue-stale --minutes 20
    python -m app.engine.bridge_admin purge

Queue root is ``MEMORY_BRIDGE_DIR`` (default ``<repo>/.bridge``), matching
``BridgeLLMClient``.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# app/engine/bridge_admin.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

WORKER_PROTOCOL = """\
Bridge worker contract
=======================

The bridge is a three-directory file queue under MEMORY_BRIDGE_DIR
(default <repo>/.bridge):

    pending/   req_<id>.json   — published requests, awaiting a worker
    claimed/   req_<id>.json   — a worker has taken ownership
    done/      <id>.json       — the worker's answer

A worker loops:

1. CLAIM. Pick a file ``pending/req_<id>.json`` and take ownership by
   RENAMING it to ``claimed/req_<id>.json``. Rename is atomic on a local
   filesystem, so exactly one worker wins a race for a given file; a worker
   that loses the rename simply picks another. Never process a file while it is
   still in pending/.

2. READ. The claimed file is JSON:
       {
         "id":              "<hex id>",
         "model_size":      "small" | "medium",
         "response_schema": <JSON Schema for the required answer, or null>,
         "messages":        [ {"role": ..., "content": ...}, ... ],
         "repair_of":       "<prior id>"   # present only on repair retries
       }
   The final message already contains the response schema inline (graphiti
   appends it); ``response_schema`` is duplicated at top level so the worker can
   self-validate. A ``repair_of`` field means a previous answer failed
   validation and the appended message states the error — return corrected JSON.

3. ANSWER. Produce a JSON OBJECT that satisfies ``response_schema``. Emit raw
   JSON only — no prose, no markdown code fences. (The client has a repair
   ladder that tolerates fences/trailing-commas, but clean output avoids a retry
   round-trip.)

4. WRITE. Write the answer ATOMICALLY to ``done/<id>.json`` — write a temp file
   in done/ and rename it into place — so the client never reads a half-written
   file. Use the bare ``<id>.json`` name (no ``req_`` prefix). Leave the claimed
   file in place (or delete it); ``done/`` is what the client polls.

A request whose worker dies mid-flight is left stranded in claimed/. The
``requeue-stale`` command moves such files back to pending/ for another worker.
"""

_QUEUE_DIRS = ("pending", "claimed", "done")


def _bridge_root() -> Path:
    return Path(os.environ.get("MEMORY_BRIDGE_DIR", str(REPO_ROOT / ".bridge")))


def _json_files(directory: Path) -> list[Path]:
    """Real queue files only — skip in-flight ``.tmp_*`` writes and dotfiles."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.glob("*.json") if not p.name.startswith(".")
    )


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def cmd_status(root: Path) -> int:
    now = time.time()
    print(f"bridge: {root}")
    for name in _QUEUE_DIRS:
        directory = root / name
        files = _json_files(directory)
        if not files:
            print(f"  {name:<8} 0")
            continue
        ages = [now - f.stat().st_mtime for f in files]
        print(
            f"  {name:<8} {len(files):<4} "
            f"oldest={_fmt_age(max(ages))} newest={_fmt_age(min(ages))}"
        )
    return 0


def cmd_requeue_stale(root: Path, minutes: float) -> int:
    """Move claimed files older than ``minutes`` back to pending/ (dead worker)."""
    claimed = root / "claimed"
    pending = root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - minutes * 60
    moved = 0
    for path in _json_files(claimed):
        if path.stat().st_mtime <= cutoff:
            path.rename(pending / path.name)
            moved += 1
    print(f"requeued {moved} stale claim(s) older than {minutes:g}m")
    return 0


def cmd_purge(root: Path) -> int:
    """Delete every queued request/answer across all three dirs."""
    removed = 0
    for name in _QUEUE_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.glob("*"):
            if path.is_file():
                path.unlink()
                removed += 1
    print(f"purged {removed} file(s) from {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.engine.bridge_admin",
        description="Operate the LLM bridge file queue.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="counts + ages per queue dir")
    requeue = sub.add_parser(
        "requeue-stale", help="move claimed files older than N minutes to pending"
    )
    requeue.add_argument("--minutes", type=float, required=True)
    sub.add_parser("purge", help="clear all queue dirs")

    args = parser.parse_args(argv)
    root = _bridge_root()

    if args.command == "status":
        return cmd_status(root)
    if args.command == "requeue-stale":
        return cmd_requeue_stale(root, args.minutes)
    if args.command == "purge":
        return cmd_purge(root)
    parser.error(f"unknown command {args.command!r}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
