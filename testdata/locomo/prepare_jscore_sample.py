"""Freeze a deterministic, category-stratified LoCoMo J-score sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from testdata.locomo.run_locomo import DATASET_SHA256, file_sha256

DEFAULT_SEED = "locomo-jscore-third-v1"
CATEGORIES = (1, 2, 3, 4)


def qid_for(sample_id: str, index: int) -> str:
    return f"{sample_id}/q{index}"


def select_qids(
    dataset: list[dict], seed: str = DEFAULT_SEED
) -> tuple[list[str], dict[int, int], dict[int, int]]:
    """Choose floor(N/3) per category by a stable SHA-256 ordering."""
    by_category: dict[int, list[str]] = defaultdict(list)
    dataset_order: list[str] = []
    for sample in dataset:
        sample_id = str(sample["sample_id"])
        for index, qa in enumerate(sample.get("qa", [])):
            category = int(qa["category"])
            if category not in CATEGORIES:
                continue
            qid = qid_for(sample_id, index)
            by_category[category].append(qid)
            dataset_order.append(qid)

    selected: set[str] = set()
    populations: dict[int, int] = {}
    sample_counts: dict[int, int] = {}
    for category in CATEGORIES:
        qids = by_category[category]
        populations[category] = len(qids)
        ranked = sorted(
            qids,
            key=lambda qid: hashlib.sha256(
                f"{seed}\0{qid}".encode("utf-8")
            ).digest(),
        )
        take = len(ranked) // 3
        sample_counts[category] = take
        selected.update(ranked[:take])

    return [qid for qid in dataset_order if qid in selected], populations, sample_counts


def build_manifest(dataset_path: Path, seed: str = DEFAULT_SEED) -> dict:
    if file_sha256(dataset_path) != DATASET_SHA256:
        raise ValueError("dataset does not match the pinned official LoCoMo10 release")
    dataset = json.loads(dataset_path.read_text())
    qids, populations, sample_counts = select_qids(dataset, seed)
    if Counter(qid.split("/q", 1)[0] for qid in qids).keys() != {
        str(sample["sample_id"]) for sample in dataset
    }:
        raise ValueError("sample unexpectedly omits a complete conversation")
    qid_digest = hashlib.sha256("\n".join(qids).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "benchmark": "LoCoMo10",
        "dataset_sha256": DATASET_SHA256,
        "eligible_categories": list(CATEGORIES),
        "excluded_categories": [5],
        "fraction": {"numerator": 1, "denominator": 3},
        "selection": "lowest SHA-256(seed + NUL + qid) per category; floor(N/3)",
        "seed": seed,
        "category_population": {str(key): value for key, value in populations.items()},
        "category_sample": {str(key): value for key, value in sample_counts.items()},
        "question_count": len(qids),
        "qid_list_sha256": qid_digest,
        "qids": qids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing != manifest:
            raise SystemExit("existing sample manifest differs; choose a new output path")
    else:
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output)
    print(
        f"wrote {manifest['question_count']} frozen qids to {args.output} "
        f"(sha256={manifest['qid_list_sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
