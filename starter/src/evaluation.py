"""Evaluate source-and-evidence hits while preserving full retrieval traces."""
import re
import unicodedata
from typing import Any


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Mark a hit only when a gold-page chunk contains the full gold evidence."""
    required = ("id", "split", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote")
    missing = [key for key in required if key not in question]
    if missing:
        raise ValueError(f"Benchmark question is missing fields: {missing}")

    evidence = _normalize_for_match(question["evidence_quote"])
    if not evidence:
        raise ValueError(f"{question['id']} has empty evidence_quote")

    traced_hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, 1):
        source_page_match = (
            hit.get("document_id") == question["gold_source_id"]
            and hit.get("page") == question["gold_page"]
        )
        evidence_match = evidence in _normalize_for_match(hit.get("text", ""))
        traced_hits.append(
            {
                "rank": rank,
                **hit,
                "source_page_match": source_page_match,
                "evidence_match": evidence_match,
                "evidence_hit": source_page_match and evidence_match,
            }
        )

    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "filter": question.get("filter"),
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "retrieved": traced_hits,
        "hit_at_1": bool(traced_hits and traced_hits[0]["evidence_hit"]),
        "evidence_hit_at_3": any(hit["evidence_hit"] for hit in traced_hits[:3]),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for trace in traces:
        strategy = trace["strategy"]
        split = trace["split"]
        grouped.setdefault(strategy, {}).setdefault(split, []).append(trace)

    summary: dict[str, Any] = {}
    for strategy, splits in sorted(grouped.items()):
        summary[strategy] = {}
        for split, rows in sorted(splits.items()):
            total = len(rows)
            hit1 = sum(bool(row["hit_at_1"]) for row in rows)
            hit3 = sum(bool(row["evidence_hit_at_3"]) for row in rows)
            summary[strategy][split] = {
                "questions": total,
                "hit_at_1": {"count": hit1, "total": total, "rate": hit1 / total},
                "evidence_hit_at_3": {
                    "count": hit3,
                    "total": total,
                    "rate": hit3 / total,
                },
            }
    return summary
