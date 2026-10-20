"""Run Mem0-style LoCoMo answer generation and binary LLM judging.

The answer phase reads only ``blind.jsonl``. The judge phase runs later and
reads gold answers from ``retrieval.jsonl`` plus the frozen predictions. Each
question is handled by an isolated Codex invocation to prevent cross-question
leakage. No prior run directory is consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from testdata.locomo.mem0_prompts_4b61c5d import (
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    preprocess_answer,
)

MEM0_BENCHMARK_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
PROMPT_SNAPSHOT_SHA256 = "8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2"
REASONING_EFFORT = "low"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def verify_prompt_snapshot() -> None:
    prompt_path = Path(__file__).with_name("mem0_prompts_4b61c5d.py")
    actual = file_sha256(prompt_path)
    if actual != PROMPT_SNAPSHOT_SHA256:
        raise RuntimeError(
            f"Mem0 prompt snapshot changed: expected {PROMPT_SNAPSHOT_SHA256}, got {actual}"
        )


def answer_prompt(record: dict) -> str:
    return get_answer_generation_prompt(
        record["question"],
        list(record.get("search_results") or [])[:50],
        reference_date=record.get("reference_date"),
    )


def judge_prompt(record: dict, generated_answer: str) -> str:
    gold = preprocess_answer(int(record["category"]), str(record["answer"]))
    user_prompt = get_judge_prompt(
        int(record["category"]), record["question"], gold, generated_answer
    )
    # Codex CLI has no separate system-message flag. Preserve the exact Mem0
    # system and user text in their original order in one isolated request.
    return f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"


def output_stem(phase: str, qid: str) -> str:
    digest = hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16]
    return f"{phase}_{digest}"


def tool_event_count(log_text: str) -> int:
    count = 0
    for line in log_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item") if isinstance(event, dict) else None
        kind = item.get("type", "") if isinstance(item, dict) else ""
        if kind in {
            "command_execution",
            "mcp_tool_call",
            "web_search",
            "file_change",
            "collab_agent_tool_call",
        }:
            count += 1
    return count


def run_codex(
    *,
    phase: str,
    qid: str,
    prompt: str,
    output_dir: Path,
    schema_path: Path,
    model: str,
    attempts: int,
) -> dict:
    stem = output_stem(phase, qid)
    output_path = output_dir / f"{stem}.json"
    metadata_path = output_dir / f"{stem}.meta.json"
    log_path = output_dir / f"{stem}.jsonl.log"
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "phase": phase,
                "qid": qid,
                "prompt": prompt,
                "model": model,
                "reasoning_effort": REASONING_EFFORT,
                "prompt_snapshot": PROMPT_SNAPSHOT_SHA256,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("cache_key") == cache_key:
            return json.loads(output_path.read_text())

    output_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    command = [
        "codex",
        "exec",
        "-m",
        model,
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ignore-rules",
        "--json",
        "-c",
        f'model_reasoning_effort="{REASONING_EFFORT}"',
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            process = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=tempfile.gettempdir(),
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"Codex call timed out after {exc.timeout}s"
            output_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 * attempt)
            continue
        log_path.write_text(process.stdout)
        if process.returncode == 0 and output_path.exists():
            try:
                result = json.loads(output_path.read_text())
                atomic_json(
                    metadata_path,
                    {
                        "cache_key": cache_key,
                        "qid": qid,
                        "model": model,
                        "reasoning_effort": REASONING_EFFORT,
                        "tool_events": tool_event_count(process.stdout),
                    },
                )
                return result
            except json.JSONDecodeError as exc:
                last_error = f"invalid JSON output: {exc}"
        else:
            last_error = process.stdout[-2000:]
        output_path.unlink(missing_ok=True)
        if attempt < attempts:
            time.sleep(2 * attempt)
    raise RuntimeError(f"{phase} call for {qid} failed: {last_error}")


def run_parallel(
    records: list[dict],
    worker,
    *,
    phase: str,
    concurrency: int,
) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, record): record["qid"] for record in records}
        completed = 0
        for future in as_completed(futures):
            qid = futures[future]
            outputs[qid] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(records):
                print(f"{phase}: {completed}/{len(records)}", flush=True)
    return outputs


def answer_phase(run_dir: Path, model: str, concurrency: int, attempts: int) -> None:
    blind_path = run_dir / "blind.jsonl"
    blind = read_jsonl(blind_path)
    if not blind:
        raise RuntimeError("blind.jsonl is empty")
    if len({record["qid"] for record in blind}) != len(blind):
        raise RuntimeError("blind.jsonl contains duplicate qids")
    output_dir = run_dir / "jscore_answer_calls"
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "schema.json"
    atomic_json(
        schema_path,
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    def worker(record: dict) -> dict:
        return run_codex(
            phase="answer",
            qid=record["qid"],
            prompt=answer_prompt(record),
            output_dir=output_dir,
            schema_path=schema_path,
            model=model,
            attempts=attempts,
        )

    outputs = run_parallel(blind, worker, phase="answer", concurrency=concurrency)
    predictions = []
    for record in blind:
        answer = str(outputs[record["qid"]]["answer"]).strip()
        if "ANSWER:" in answer:
            answer = answer.rsplit("ANSWER:", 1)[-1].strip()
        predictions.append({"qid": record["qid"], "answer": answer, "model": model})
    atomic_jsonl(run_dir / "predictions.jsonl", predictions)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [100 * (center - radius), 100 * (center + radius)]


def judge_phase(run_dir: Path, model: str, concurrency: int, attempts: int) -> None:
    private_records = read_jsonl(run_dir / "retrieval.jsonl")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    prediction_by_qid = {row["qid"]: row["answer"] for row in predictions}
    answerer_models = {row.get("model") for row in predictions}
    if len(answerer_models) != 1 or None in answerer_models:
        raise RuntimeError("predictions must contain exactly one answerer model")
    answerer_model = answerer_models.pop()
    expected_qids = [row["qid"] for row in private_records]
    if (
        set(expected_qids) != set(prediction_by_qid)
        or len(predictions) != len(private_records)
    ):
        raise RuntimeError("prediction qids do not exactly match retrieval qids")

    output_dir = run_dir / "jscore_judge_calls"
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / "schema.json"
    atomic_json(
        schema_path,
        {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "label": {"type": "string", "enum": ["CORRECT", "WRONG"]},
            },
            "required": ["reasoning", "label"],
            "additionalProperties": False,
        },
    )

    def worker(record: dict) -> dict:
        return run_codex(
            phase="judge",
            qid=record["qid"],
            prompt=judge_prompt(record, prediction_by_qid[record["qid"]]),
            output_dir=output_dir,
            schema_path=schema_path,
            model=model,
            attempts=attempts,
        )

    outputs = run_parallel(
        private_records, worker, phase="judge", concurrency=concurrency
    )
    evaluations: list[dict] = []
    by_category: dict[int, list[bool]] = defaultdict(list)
    for record in private_records:
        raw = outputs[record["qid"]]
        correct = raw["label"] == "CORRECT"
        category = int(record["category"])
        by_category[category].append(correct)
        evaluations.append(
            {
                "qid": record["qid"],
                "category": category,
                "question": record["question"],
                "gold_answer": preprocess_answer(category, str(record["answer"])),
                "generated_answer": prediction_by_qid[record["qid"]],
                "label": raw["label"],
                "reasoning": raw["reasoning"],
            }
        )

    total = len(evaluations)
    correct = sum(row["label"] == "CORRECT" for row in evaluations)
    metrics_by_category = {}
    for category in sorted(by_category):
        values = by_category[category]
        cat_correct = sum(values)
        metrics_by_category[str(category)] = {
            "name": CATEGORY_NAMES[category],
            "correct": cat_correct,
            "total": len(values),
            "accuracy": 100 * cat_correct / len(values),
            "wilson_95": wilson_interval(cat_correct, len(values)),
        }

    metadata_files = list((run_dir / "jscore_answer_calls").glob("*.meta.json"))
    metadata_files += list((run_dir / "jscore_judge_calls").glob("*.meta.json"))
    tool_events = sum(
        json.loads(path.read_text()).get("tool_events", 0) for path in metadata_files
    )
    results = {
        "protocol": {
            "benchmark": "LoCoMo10",
            "metric": "binary LLM J-score accuracy",
            "categories": [1, 2, 3, 4],
            "category_5_excluded": True,
            "cutoff": 50,
            "judge_evidence": False,
            "answerer_model": answerer_model,
            "judge_model": model,
            "reasoning_effort": REASONING_EFFORT,
            "mem0_benchmark_commit": MEM0_BENCHMARK_COMMIT,
            "prompt_snapshot_sha256": PROMPT_SNAPSHOT_SHA256,
        },
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": 100 * correct / total if total else 0.0,
            "wilson_95": wilson_interval(correct, total),
        },
        "by_category": metrics_by_category,
        "blindness_audit": {
            "answer_input": "blind.jsonl only",
            "judge_input": "question + preprocessed gold + generated answer; no evidence/context",
            "model_tool_events": tool_events,
            "blind_sha256": file_sha256(run_dir / "blind.jsonl"),
            "retrieval_sha256": file_sha256(run_dir / "retrieval.jsonl"),
            "predictions_sha256": file_sha256(run_dir / "predictions.jsonl"),
        },
        "evaluations": evaluations,
    }
    atomic_json(run_dir / "jscore_results.json", results)
    print(
        f"J-score accuracy: {correct}/{total} = {results['overall']['accuracy']:.2f}%",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--phase", choices=("answer", "judge", "all"), default="all")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    verify_prompt_snapshot()
    if args.phase in ("answer", "all"):
        answer_phase(run_dir, args.model, args.concurrency, args.attempts)
    if args.phase in ("judge", "all"):
        judge_phase(run_dir, args.model, args.concurrency, args.attempts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
