"""Resumable LoCoMo driver for the current Graphiti-native memory engine.

The official dataset contains 5,882 actor turns.  Production ingestion stores
every turn as a Graphiti message episode, which would require roughly 47,000
structured bridge responses for a full run.  This benchmark therefore uses one
dated *session* per native Graphiti message episode (272 total) and labels the
result as a session-episode ablation.  It still exercises Graphiti extraction,
dedupe/invalidation, native hybrid search, local embeddings/reranking, source
hydration, raw-passage retrieval, and the hard token budget.

Gold answers/evidence never enter ingestion or retrieval.  The driver writes:

    manifest.json       immutable run configuration and dataset pin
    state.json          tenant/session progress (resumable)
    retrieval.jsonl     private records including gold/evidence for scoring
    blind.jsonl         answer-worker input: qid/question/context only

Run from the repository root.  Environment must be set before this module is
started because graph paths and LLM mode are process-global.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DATASET_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
_SESSION_KEY = re.compile(r"^session_(\d+)$")
_DATE = re.compile(
    r"^(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$",
    re.IGNORECASE,
)
_WRITE_LOCK = threading.Lock()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{uuid.uuid4().hex}_{path.name}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{uuid.uuid4().hex}_{path.name}")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_locomo_date(raw: str) -> datetime:
    match = _DATE.fullmatch((raw or "").strip())
    if not match:
        raise ValueError(f"unparseable LoCoMo session date: {raw!r}")
    hour, minute, ampm, day, month, year = match.groups()
    hour_int = int(hour) % 12 + (12 if ampm.casefold() == "pm" else 0)
    month_int = datetime.strptime(month[:3], "%b").month
    return datetime(
        int(year), month_int, int(day), hour_int, int(minute), tzinfo=timezone.utc
    )


def session_keys(conversation: dict) -> list[str]:
    keys = [key for key in conversation if _SESSION_KEY.fullmatch(key)]
    return sorted(keys, key=lambda key: int(_SESSION_KEY.fullmatch(key).group(1)))


def turn_text(turn: dict) -> str:
    text = str(turn.get("text") or "").strip()
    caption = str(turn.get("blip_caption") or "").strip()
    if caption:
        text = f"{text} [Shared image: {caption}]" if text else f"[Shared image: {caption}]"
    return text


def render_session(conversation: dict, key: str) -> tuple[str, list[str]]:
    speaker_a = str(conversation["speaker_a"])
    lines: list[str] = []
    dia_ids: list[str] = []
    for turn in conversation[key]:
        content = turn_text(turn)
        if not content:
            continue
        role = "User" if turn.get("speaker") == speaker_a else "Assistant"
        speaker = str(turn.get("speaker") or role)
        lines.append(f"{role}: [{speaker}] {content}")
        dia_ids.append(str(turn["dia_id"]))
    if not lines:
        raise ValueError(f"{key} has no usable turns")
    return "\n".join(lines), dia_ids


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("official LoCoMo file must be a JSON list")
    return data


def load_qids(path: Path | None) -> set[str] | None:
    """Load a frozen question sample without exposing gold data to workers."""
    if path is None:
        return None
    data = json.loads(path.read_text())
    qids = data.get("qids") if isinstance(data, dict) else data
    if not isinstance(qids, list) or not qids:
        raise ValueError("qid file must be a non-empty JSON list or contain a qids list")
    if not all(isinstance(qid, str) and qid for qid in qids):
        raise ValueError("every qid must be a non-empty string")
    if len(qids) != len(set(qids)):
        raise ValueError("qid file contains duplicate qids")
    return set(qids)


def conversation_reference_date(conversation: dict) -> str | None:
    """Return the latest LoCoMo session date using Mem0's convention."""
    dates = [
        (
            parse_locomo_date(conversation[f"{key}_date_time"]),
            conversation[f"{key}_date_time"],
        )
        for key in session_keys(conversation)
    ]
    return max(dates, default=(None, None), key=lambda value: value[0])[1]


def ensure_manifest(run_dir: Path, dataset: Path, args) -> dict:
    ingestion = (
        "direct raw actor-turn episodes; Graphiti extraction bypassed"
        if args.direct_raw
        else "one Graphiti EpisodeType.message episode per LoCoMo session"
    )
    manifest = {
        "schema_version": 1,
        "benchmark": "LoCoMo10",
        "dataset_commit": DATASET_COMMIT,
        "dataset_sha256": file_sha256(dataset),
        "implementation_sha256": hashlib.sha256(b"".join(
            path.read_bytes()
            for path in (
                Path(__file__),
                REPO / "app/engine/retrieval.py",
                REPO / "app/engine/clients.py",
            )
        )).hexdigest(),
        "ingestion": ingestion,
        "production_deviation": (
            "Graphiti extraction/traversal bypassed; measures source/RAG retrieval only"
            if args.direct_raw
            else "production /v1/remember ingests one episode per actor turn"
        ),
        "token_budget": args.token_budget,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "answerdotai/answerai-colbert-small-v1",
        "retrieval_env": {
            key: os.environ.get(key)
            for key in (
                "MEMORY_SEARCH_LIMIT",
                "MEMORY_TOP_ENTITIES",
                "MEMORY_FINAL_EDGE_CAP",
                "MEMORY_RAW_EPISODE_SCAN_LIMIT",
                "MEMORY_RAW_RERANK_POOL",
                "MEMORY_RAW_FINAL_CAP",
                "MEMORY_RAW_PER_SOURCE",
                "MEMORY_RERANK_PASSAGE_CACHE",
            )
        },
        "include_historical": True,
        "llm_mode": "none-direct-raw" if args.direct_raw else "scripted-empty" if args.scripted else os.environ.get(
            "MEMORY_GRAPHITI_LLM", "replay"
        ),
        "graph_dir": os.environ.get("MEMORY_GRAPH_DIR", ".graphdb-locomo"),
        "bridge_dir": os.environ.get("MEMORY_BRIDGE_DIR", ".bridge-locomo"),
        "sample_ids": args.sample_id or "all",
        "qid_filter_sha256": file_sha256(args.qid_file) if args.qid_file else None,
        "qid_filter_count": len(load_qids(args.qid_file) or ()),
        "max_sessions": args.max_sessions,
        "max_questions": args.max_questions,
    }
    path = run_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise RuntimeError(
                "run manifest differs from requested configuration; choose a new --run-id"
            )
        return existing
    if manifest["dataset_sha256"] != DATASET_SHA256:
        raise RuntimeError(
            f"dataset SHA-256 {manifest['dataset_sha256']} does not match pinned "
            f"official file {DATASET_SHA256}"
        )
    atomic_json(path, manifest)
    return manifest


def create_tenant() -> dict[str, str]:
    from sqlalchemy import text
    from app import db

    tenant_id = str(uuid.uuid4())
    api_key = f"locomo-{uuid.uuid4().hex}"
    key_hash = hashlib.sha256(api_key.encode("utf-8")).digest()
    with db.get_admin_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO api_keys (key_hash, key_prefix, tenant_id, name, scopes)"
                " VALUES (:h, :p, :t, 'locomo',"
                " ARRAY['memory:read', 'memory:write']::text[])"
            ),
            {"h": key_hash, "p": api_key[:16], "t": tenant_id},
        )
        conn.execute(
            text(
                "INSERT INTO tenant_settings (tenant_id, timezone, locale) "
                "VALUES (:t,'UTC','en')"
            ),
            {"t": tenant_id},
        )
    return {"tenant_id": tenant_id, "api_key": api_key, "sessions": []}


def configure_engine(scripted: bool) -> None:
    from app.engine import graph
    from app.engine.clients import FastEmbedEmbedder, LateInteractionReranker

    if scripted:
        from tests.engine_fakes import ScriptedLLMClient

        llm = ScriptedLLMClient()
    else:
        from app.engine.bridge import get_llm_client

        llm = get_llm_client()
    embedder = FastEmbedEmbedder()
    graph.configure_clients(llm, embedder, LateInteractionReranker())

    # Load ONNX weights once on the main thread; the client's lazy loader is not
    # intended to race during the first concurrent ingestion.
    import asyncio

    asyncio.run(embedder.create(["LoCoMo warmup"]))


def ensure_session(tenant_id: str, sample_id: str, conversation: dict, key: str) -> str:
    from sqlalchemy import text as sql_text
    from app.db import tenant_connection
    from app.engine.ingest import (
        episode_in_graph,
        ingest_episode_into_graph,
        ingest_text,
    )
    from app.episodes import canonical_hash

    body, dia_ids = render_session(conversation, key)
    occurred_at = parse_locomo_date(conversation[f"{key}_date_time"])
    meta = {
        "benchmark": "locomo",
        "sample_id": sample_id,
        "session_id": key,
        "dia_ids": dia_ids,
        "speaker_a": conversation["speaker_a"],
        "speaker_b": conversation["speaker_b"],
        "role": "session",
    }
    digest = canonical_hash(body, None)
    with tenant_connection(tenant_id) as conn:
        row = conn.execute(
            sql_text(
                "SELECT episode_id FROM episodes "
                "WHERE tenant_id=:tenant AND content_hash=:digest"
            ),
            {"tenant": tenant_id, "digest": digest},
        ).first()

    if row is None:
        report = ingest_text(
            tenant_id, text=body, occurred_at=occurred_at, source="chat", meta=meta
        )
        return report.episode_id

    episode_id = str(row.episode_id)
    if not episode_in_graph(tenant_id, episode_id):
        ingest_episode_into_graph(
            tenant_id,
            episode_id,
            text=body,
            occurred_at=occurred_at,
            source="chat",
            meta=meta,
        )
    return episode_id


def ensure_session_raw(tenant_id: str, sample_id: str, conversation: dict, key: str) -> list[str]:
    """Store each LoCoMo actor turn directly in the immutable episode ledger.

    This deliberately bypasses Graphiti extraction and is therefore a source/RAG
    baseline, not a graph-memory score. It otherwise matches production's
    actor-turn granularity and metadata closely.
    """
    from datetime import timedelta
    from app.db import tenant_connection
    from app.episodes import ingest_episode

    occurred_at = parse_locomo_date(conversation[f"{key}_date_time"])
    speaker_a = str(conversation["speaker_a"])
    episode_ids: list[str] = []
    with tenant_connection(tenant_id) as conn:
        for index, turn in enumerate(conversation[key]):
            content = turn_text(turn)
            if not content:
                continue
            role = "user" if turn.get("speaker") == speaker_a else "assistant"
            speaker = str(turn.get("speaker") or role)
            dia_id = str(turn["dia_id"])
            episode_id, _ = ingest_episode(
                conn,
                tenant_id,
                source="chat",
                occurred_at=occurred_at + timedelta(microseconds=index),
                content=f"{role}: [{speaker}] {content}",
                # Benchmark turns have distinct evidence identities even when
                # two speakers happen to repeat the same short farewell. Keep
                # the production content unchanged while preventing the
                # ledger's content-only dedupe from collapsing dia_ids.
                content_hash=hashlib.sha256(
                    f"locomo:{sample_id}:{dia_id}\n{content}".encode("utf-8")
                ).hexdigest(),
                source_ref=f"locomo:{sample_id}:{dia_id}",
                meta={
                    "benchmark": "locomo",
                    "sample_id": sample_id,
                    "session_id": key,
                    "dia_ids": [dia_id],
                    "speaker": speaker,
                    "role": role,
                    "turn_index": index,
                },
            )
            episode_ids.append(str(episode_id))
    return episode_ids


def ingest_sample(
    sample: dict,
    state: dict,
    state_path: Path,
    max_sessions: int | None,
    direct_raw: bool,
) -> None:
    sample_id = str(sample["sample_id"])
    with _WRITE_LOCK:
        tenant = state["tenants"].get(sample_id)
        if tenant is None:
            tenant = create_tenant()
            state["tenants"][sample_id] = tenant
        atomic_json(state_path, state)
    completed = set(tenant.get("sessions", []))
    keys = session_keys(sample["conversation"])
    if max_sessions is not None:
        keys = keys[:max_sessions]
    for index, key in enumerate(keys, start=1):
        if key in completed:
            continue
        if direct_raw:
            ensure_session_raw(tenant["tenant_id"], sample_id, sample["conversation"], key)
        else:
            ensure_session(tenant["tenant_id"], sample_id, sample["conversation"], key)
        with _WRITE_LOCK:
            tenant["sessions"].append(key)
            atomic_json(state_path, state)
        print(f"ingest {sample_id}: {index}/{len(keys)} {key}", flush=True)


def build_episode_maps(tenant_id: str) -> tuple[dict[str, dict], dict[str, str]]:
    from sqlalchemy import text as sql_text
    from app.db import tenant_connection
    from app.engine import graph

    with tenant_connection(tenant_id) as conn:
        rows = conn.execute(
            sql_text(
                "SELECT episode_id, meta FROM episodes "
                "WHERE tenant_id=:tenant AND meta->>'benchmark'='locomo'"
            ),
            {"tenant": tenant_id},
        ).mappings().all()
    pg = {str(row.episode_id): dict(row.meta or {}) for row in rows}

    engine = graph.get_engine(tenant_id)

    async def graph_names():
        records, _, _ = await engine.driver.execute_query(
            "MATCH (e:Episodic) RETURN e.uuid AS uuid, e.name AS name"
        )
        return {str(record["uuid"]): str(record["name"]) for record in records}

    return pg, graph._run_coro(graph_names())


def evidence_from_items(items: list[dict], pg: dict[str, dict], graph_names: dict[str, str]) -> set[str]:
    """Resolve packed raw passages, graph episodes, and fact provenance to dia_ids.

    Raw passage suffixes map one-to-one to session turns because every released
    LoCoMo turn (including its BLIP caption) is shorter than the 900-character
    passage chunk size. Graph episode items credit the full session, matching the
    official session-unit retrieval metric.
    """
    retrieved: set[str] = set()
    for item in items:
        episode_ids = []
        if item.get("episode_id"):
            episode_ids.append(item["episode_id"])
        episode_ids.extend(item.get("episode_ids") or [])
        for episode_id in episode_ids:
            pg_id = str(episode_id)
            meta = pg.get(pg_id)
            if meta is None:
                pg_id = graph_names.get(pg_id, "")
                meta = pg.get(pg_id)
            if not meta:
                continue
            dia_ids = list(meta.get("dia_ids") or [])
            passage_id = item.get("passage_id")
            if passage_id and str(passage_id).startswith(f"{pg_id}:"):
                try:
                    index = int(str(passage_id).rsplit(":", 1)[1])
                except ValueError:
                    continue
                if 0 <= index < len(dia_ids):
                    retrieved.add(str(dia_ids[index]))
                # An invalid passage suffix credits no evidence, never the
                # whole parent episode.
                continue
            retrieved.update(str(dia_id) for dia_id in dia_ids)
    return retrieved


def assert_ingestion_complete(sample: dict, tenant: dict, max_sessions: int | None) -> None:
    expected = session_keys(sample["conversation"])
    if max_sessions is not None:
        expected = expected[:max_sessions]
    completed = list(tenant.get("sessions") or [])
    if set(completed) != set(expected) or len(completed) != len(expected):
        missing = sorted(set(expected) - set(completed))
        extra = sorted(set(completed) - set(expected))
        raise RuntimeError(
            f"{sample['sample_id']} ingestion incomplete: missing={missing}, extra={extra}"
        )


def retrieve_sample(
    sample: dict,
    tenant: dict,
    token_budget: int,
    wanted_qids: set[str] | None = None,
) -> list[dict]:
    from app.db import tenant_connection
    from app.engine.retrieval import assemble_context

    sample_id = str(sample["sample_id"])
    tenant_id = tenant["tenant_id"]
    pg, graph_names = build_episode_maps(tenant_id)
    records: list[dict] = []
    for index, qa in enumerate(sample.get("qa", [])):
        qid = f"{sample_id}/q{index}"
        if wanted_qids is not None and qid not in wanted_qids:
            continue
        with tenant_connection(tenant_id) as conn:
            response = assemble_context(
                conn,
                tenant_id,
                query=qa["question"],
                token_budget=token_budget,
                include_historical=True,
                fmt="json",
                actor="benchmark.locomo",
            )
        items = list(response.get("items") or [])
        context = "\n".join(item.get("line", "") for item in items if item.get("line"))
        retrieved = sorted(evidence_from_items(items, pg, graph_names))
        records.append(
            {
                "qid": qid,
                "sample_id": sample_id,
                "question": qa["question"],
                "answer": qa.get("answer"),
                "adversarial_answer": qa.get("adversarial_answer"),
                "category": int(qa["category"]),
                "category_name": CATEGORY_NAMES[int(qa["category"])],
                "evidence": list(qa.get("evidence") or []),
                "retrieved_evidence": retrieved,
                "context": context,
                "context_items": items,
                "tokens": response.get("tokens_estimated", 0),
                "route": response.get("route"),
                "reference_date": conversation_reference_date(sample["conversation"]),
            }
        )
    print(f"retrieve {sample_id}: {len(records)} questions", flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=REPO / "testdata/locomo/data/locomo10.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=("all", "ingest", "retrieve"), default="all")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument(
        "--qid-file",
        type=Path,
        help="frozen JSON list (or object with qids) selecting exact questions",
    )
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument(
        "--direct-raw",
        action="store_true",
        help="store actor turns directly; bypass Graphiti extraction (source/RAG baseline)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        parser.error(f"dataset not found: {args.dataset}")
    if args.qid_file and not args.qid_file.exists():
        parser.error(f"qid file not found: {args.qid_file}")
    if args.qid_file and args.max_questions is not None:
        parser.error("--qid-file and --max-questions cannot be combined")
    try:
        wanted_qids = load_qids(args.qid_file)
    except ValueError as exc:
        parser.error(str(exc))
    run_dir = REPO / "testdata/locomo/runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(run_dir, args.dataset, args)
    samples = load_rows(args.dataset)
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [sample for sample in samples if str(sample["sample_id"]) in wanted]
        missing = wanted - {str(sample["sample_id"]) for sample in samples}
        if missing:
            parser.error(f"unknown sample id(s): {sorted(missing)}")
    if wanted_qids is not None:
        available_qids = {
            f"{sample['sample_id']}/q{index}"
            for sample in samples
            for index, _ in enumerate(sample.get("qa", []))
        }
        missing_qids = wanted_qids - available_qids
        if missing_qids:
            parser.error(f"unknown qid(s): {sorted(missing_qids)[:10]}")

    # Import/migrate only after immutable path + LLM environment was selected.
    from alembic import command
    from alembic.config import Config
    from app.engine import graph

    cfg = Config(str(REPO / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "alembic"))
    command.upgrade(cfg, "head")
    configure_engine(args.scripted or args.direct_raw)

    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"tenants": {}}

    try:
        if args.phase in ("all", "ingest"):
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        ingest_sample,
                        sample,
                        state,
                        state_path,
                        args.max_sessions,
                        args.direct_raw,
                    ): sample
                    for sample in samples
                }
                for future in as_completed(futures):
                    future.result()

        if args.phase in ("all", "retrieve"):
            for sample in samples:
                sample_id = str(sample["sample_id"])
                if sample_id not in state["tenants"]:
                    raise RuntimeError(f"{sample_id} has not been ingested")
                assert_ingestion_complete(
                    sample, state["tenants"][sample_id], args.max_sessions
                )
            sample_dir = run_dir / "retrieval_samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            records_by_sample: dict[str, list[dict]] = {}
            todo_samples = []
            for sample in samples:
                sample_id = str(sample["sample_id"])
                sample_path = sample_dir / f"{sample_id}.jsonl"
                if sample_path.exists():
                    records_by_sample[sample_id] = [
                        json.loads(line)
                        for line in sample_path.read_text().splitlines()
                        if line.strip()
                    ]
                else:
                    todo_samples.append(sample)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        retrieve_sample,
                        sample,
                        state["tenants"][str(sample["sample_id"])],
                        args.token_budget,
                        wanted_qids,
                    ): str(sample["sample_id"])
                    for sample in todo_samples
                }
                for future in as_completed(futures):
                    sample_id = futures[future]
                    records_by_sample[sample_id] = future.result()
                    atomic_jsonl(
                        sample_dir / f"{sample_id}.jsonl",
                        records_by_sample[sample_id],
                    )
            all_records = [
                record
                for sample in samples
                for record in records_by_sample[str(sample["sample_id"])]
            ]
            if args.max_questions is not None:
                all_records = all_records[: args.max_questions]
            atomic_jsonl(run_dir / "retrieval.jsonl", all_records)
            blind = [
                {
                    "qid": row["qid"],
                    "question": row["question"],
                    "reference_date": row["reference_date"],
                    "search_results": [
                        {
                            "memory": (
                                item.get("snippet")
                                or item.get("fact")
                                or item.get("line", "")
                            ),
                            "created_at": item.get("occurred_at") or "",
                        }
                        for item in row["context_items"]
                    ],
                }
                for row in all_records
            ]
            atomic_jsonl(run_dir / "blind.jsonl", blind)
            print(f"wrote {len(all_records)} retrieval + blind records to {run_dir}")
    finally:
        graph.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
