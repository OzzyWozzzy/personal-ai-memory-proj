"""Score frozen LoCoMo predictions with the official ACL evaluator semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

try:
    from nltk.stem import PorterStemmer
except ImportError as exc:  # pragma: no cover - operator-facing dependency error
    raise SystemExit("score_locomo.py requires nltk: .venv/bin/pip install nltk") from exc

STEMMER = PorterStemmer()
ARTICLES = re.compile(r"\b(a|an|the|and)\b")
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: object) -> str:
    value = str(text or "").replace(",", "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = ARTICLES.sub(" ", value)
    return " ".join(value.split())


def token_f1(prediction: object, ground_truth: object) -> float:
    predicted = [STEMMER.stem(token) for token in normalize(prediction).split()]
    gold = [STEMMER.stem(token) for token in normalize(ground_truth).split()]
    if not predicted or not gold:
        return 0.0
    same = sum((Counter(predicted) & Counter(gold)).values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def official_f1(prediction: str, ground_truth: object, category: int) -> float:
    if category == 5:
        lower = prediction.lower()
        return float("no information available" in lower or "not mentioned" in lower)
    gold = str(ground_truth or "")
    if category == 3:
        gold = gold.split(";", 1)[0].strip()
    if category == 1:
        predictions = [part.strip() for part in prediction.split(",")]
        answers = [part.strip() for part in gold.split(",")]
        return sum(max(token_f1(part, answer) for part in predictions) for answer in answers) / len(answers)
    return token_f1(prediction, gold)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    retrieval_path = args.run_dir / "retrieval.jsonl"
    predictions_path = args.predictions or args.run_dir / "predictions.jsonl"
    rows = read_jsonl(retrieval_path)
    prediction_rows = read_jsonl(predictions_path)
    prediction_qids = [row["qid"] for row in prediction_rows]
    if len(prediction_qids) != len(set(prediction_qids)):
        raise SystemExit("predictions contain duplicate qids")
    predictions = {row["qid"]: str(row["answer"]) for row in prediction_rows}
    expected_qids = {row["qid"] for row in rows}
    if set(predictions) != expected_qids:
        raise SystemExit(
            f"prediction qids differ: missing={sorted(expected_qids-set(predictions))[:5]}, "
            f"extra={sorted(set(predictions)-expected_qids)[:5]}"
        )
    missing = [row["qid"] for row in rows if row["qid"] not in predictions]
    if missing:
        raise SystemExit(f"missing {len(missing)} predictions; first: {missing[:5]}")

    scores: dict[int, list[float]] = defaultdict(list)
    recalls: dict[int, list[float]] = defaultdict(list)
    complete: dict[int, list[float]] = defaultdict(list)
    details = []
    for row in rows:
        category = int(row["category"])
        prediction = predictions[row["qid"]]
        # The official runner rounds each question before evaluation_stats.py
        # aggregates the result.
        score = round(official_f1(prediction, row.get("answer"), category), 3)
        evidence = list(row.get("evidence") or [])
        retrieved = set(row.get("retrieved_evidence") or [])
        # Keep duplicate evidence ids: the upstream evaluator iterates the list,
        # and one released question intentionally/accidentally repeats an id.
        recall = sum(item in retrieved for item in evidence) / len(evidence) if evidence else 1.0
        scores[category].append(score)
        recalls[category].append(recall)
        complete[category].append(float(all(item in retrieved for item in evidence)))
        details.append({
            "qid": row["qid"], "category": category, "prediction": prediction,
            "answer": row.get("answer"), "f1": score,
            "evidence_recall": recall,
            "complete_evidence": all(item in retrieved for item in evidence),
        })

    non_adversarial = [score for category in (1, 2, 3, 4) for score in scores[category]]
    all_scores = [score for values in scores.values() for score in values]
    result = {
        "questions": len(rows),
        "retrieval_sha256": sha256(retrieval_path),
        "predictions_sha256": sha256(predictions_path),
        "prediction_models": sorted({str(row.get("model", "unknown")) for row in prediction_rows}),
        "score_scale": "0_to_1; multiply by 100 for ACL-style tables",
        "primary_non_adversarial_f1": mean(non_adversarial),
        "primary_non_adversarial_f1_percent": 100 * mean(non_adversarial),
        "mixed_all_categories_score": mean(all_scores),
        "mixed_all_categories_score_percent": 100 * mean(all_scores),
        "adversarial_abstention": mean(scores[5]),
        "adversarial_abstention_percent": 100 * mean(scores[5]),
        "evidence_recall": mean([value for values in recalls.values() for value in values]),
        "complete_evidence_rate": mean([value for values in complete.values() for value in values]),
        "categories": {
            CATEGORIES[category]: {
                "category_id": category,
                "n": len(scores[category]),
                "f1": mean(scores[category]),
                "f1_percent": 100 * mean(scores[category]),
                "evidence_recall": mean(recalls[category]),
                "complete_evidence_rate": mean(complete[category]),
            }
            for category in sorted(scores)
        },
    }
    (args.run_dir / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.run_dir / "scored.jsonl").open("w") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
