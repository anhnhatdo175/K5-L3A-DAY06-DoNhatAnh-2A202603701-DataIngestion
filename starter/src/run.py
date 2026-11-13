"""Run the ingestion benchmark and save retrieval traces and audit sheets."""
import csv
import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import by_paragraph, fixed_size
from .embeddings import FastEmbedder
from .evaluation import evaluate_query, summarize, _normalize_for_match
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
BENCHMARK_PATH = ROOT / "benchmark" / "questions.json"
RESULTS_PATH = ROOT / "results"
MODEL_NAME = FastEmbedder.DEFAULT_MODEL
COHERENCE_SAMPLE_PAGES_PER_CHAPTER = 5

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _validate_inputs(manifest: dict[str, Any], benchmark: dict[str, Any]) -> None:
    selected_pages = manifest.get("selected_pages")
    if not isinstance(selected_pages, list) or len(selected_pages) < 30:
        raise ValueError("Manifest must list at least 30 selected content pages")
    page_numbers = [page["pdf_page"] for page in selected_pages]
    if len(page_numbers) != len(set(page_numbers)):
        raise ValueError("Manifest page IDs must be unique")
    if sum(page["normalized_characters"] for page in selected_pages) != manifest.get(
        "normalized_character_count"
    ):
        raise ValueError("Manifest page character counts do not match its total")

    questions = benchmark.get("records")
    if not isinstance(questions, list):
        raise ValueError("Benchmark records must be a list")
    if len(questions) < 20:
        raise ValueError("Benchmark must contain at least 20 questions")
    question_ids = [question.get("id") for question in questions]
    if any(not question_id for question_id in question_ids):
        raise ValueError("Every benchmark question must have an ID")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Benchmark question IDs must be unique")
    if sum(question.get("split") == "dev" for question in questions) < 15:
        raise ValueError("Benchmark must contain at least 15 dev questions")
    if sum(question.get("split") == "test" for question in questions) < 5:
        raise ValueError("Benchmark must contain at least 5 test questions")
    if not any(question.get("split") == "test" and question.get("filter") for question in questions):
        raise ValueError("Benchmark must include at least one filtered test question")
    selected_set = set(page_numbers)
    for question in questions:
        required_fields = (
            "id",
            "split",
            "question",
            "expected_answer",
            "gold_source_id",
            "gold_page",
            "evidence_quote",
        )
        if any(not question.get(field) for field in required_fields):
            raise ValueError(f"{question.get('id')} has missing benchmark fields")
        if question["split"] not in ("dev", "test"):
            raise ValueError(f"{question['id']} has unsupported split {question['split']}")
        if question.get("gold_page") not in selected_set:
            raise ValueError(f"{question.get('id')} gold page is outside the selected corpus")
        if question.get("gold_source_id") != manifest.get("source_id"):
            raise ValueError(f"{question.get('id')} gold source does not match the manifest")
        filters = question.get("filter")
        if filters and (
            not isinstance(filters, dict)
            or set(filters) != {"chapter"}
            or type(filters["chapter"]) is not int
            or filters["chapter"] not in (3, 4)
        ):
            raise ValueError(f"{question.get('id')} has an unsupported chapter filter")


def _normalized_character_count(text: str) -> int:
    return len(" ".join(text.split()))


def _chunk_stats(chunks: list[Any]) -> dict[str, Any]:
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        raise ValueError("Strategy produced no chunks")
    return {
        "chunk_count": len(chunks),
        "total_chunk_characters": sum(lengths),
        "min_chunk_characters": min(lengths),
        "mean_chunk_characters": statistics.mean(lengths),
        "max_chunk_characters": max(lengths),
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_audit_sheets(
    output_dir: Path,
    traces: list[dict[str, Any]],
    chunks_by_strategy: dict[str, list[Any]],
    pages: list[Any],
) -> list[int]:
    test_traces = [trace for trace in traces if trace["split"] == "test"]
    metadata_rows: list[dict[str, Any]] = []
    grounding_rows: list[dict[str, Any]] = []
    for trace in test_traces:
        for hit in trace["retrieved"][:3]:
            metadata_rows.append(
                {
                    "question_id": trace["question_id"],
                    "strategy": trace["strategy"],
                    "rank": hit["rank"],
                    "chunk_id": hit["chunk_id"],
                    "source": hit["source"],
                    "pdf_page": hit["page"],
                    "chapter": hit["metadata"].get("chapter"),
                    "expected_filter": json.dumps(trace["filter"], ensure_ascii=False),
                    "metadata_correct_manual": "",
                    "filter_compliant_manual": "",
                    "notes": "",
                }
            )
        grounding_rows.append(
            {
                "question_id": trace["question_id"],
                "strategy": trace["strategy"],
                "question": trace["question"],
                "gold_evidence": trace["evidence_quote"],
                "top3": json.dumps(
                    [
                        {
                            "rank": hit["rank"],
                            "chunk_id": hit["chunk_id"],
                            "score": hit["score"],
                            "source": hit["source"],
                            "pdf_page": hit["page"],
                            "text": hit["text"],
                        }
                        for hit in trace["retrieved"][:3]
                    ],
                    ensure_ascii=False,
                ),
                "grounding_score_0_1_2": "",
                "reason": "",
            }
        )

    pages_by_chapter: dict[int, list[Any]] = {}
    for page in pages:
        pages_by_chapter.setdefault(int(page.metadata["chapter"]), []).append(page)
    sample_pages: list[int] = []
    for chapter in sorted(pages_by_chapter):
        chapter_pages = sorted(pages_by_chapter[chapter], key=lambda page: page.page)
        sample_pages.extend(
            page.page for page in chapter_pages[:COHERENCE_SAMPLE_PAGES_PER_CHAPTER]
        )

    coherence_rows: list[dict[str, Any]] = []
    for page_number in sample_pages:
        page = next(page for page in pages if page.page == page_number)
        char_position = len(page.text) // 2
        for strategy, chunks in chunks_by_strategy.items():
            covering = [
                chunk
                for chunk in chunks
                if chunk.page == page_number
                and chunk.metadata["char_start"] <= char_position < chunk.metadata["char_end"]
            ]
            if not covering:
                raise RuntimeError(
                    f"No {strategy} chunk covers coherence anchor on PDF page {page_number}"
                )
            for chunk in covering:
                coherence_rows.append(
                    {
                        "anchor_id": f"pdf-{page_number}-midpoint",
                        "strategy": strategy,
                        "pdf_page": page_number,
                        "char_position": char_position,
                        "chunk_id": chunk.chunk_id,
                        "chunk_text": chunk.text,
                        "coherence_score_0_1_2": "",
                        "reason": "",
                    }
                )

    _write_csv(
        output_dir / "manual_metadata_audit.csv",
        [
            "question_id",
            "strategy",
            "rank",
            "chunk_id",
            "source",
            "pdf_page",
            "chapter",
            "expected_filter",
            "metadata_correct_manual",
            "filter_compliant_manual",
            "notes",
        ],
        metadata_rows,
    )
    _write_csv(
        output_dir / "manual_grounding_audit.csv",
        [
            "question_id",
            "strategy",
            "question",
            "gold_evidence",
            "top3",
            "grounding_score_0_1_2",
            "reason",
        ],
        grounding_rows,
    )
    _write_csv(
        output_dir / "manual_coherence_audit.csv",
        [
            "anchor_id",
            "strategy",
            "pdf_page",
            "char_position",
            "chunk_id",
            "chunk_text",
            "coherence_score_0_1_2",
            "reason",
        ],
        coherence_rows,
    )
    return sample_pages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_started_at = datetime.now(timezone.utc)
    run_id = run_started_at.strftime("%Y%m%dT%H%M%SZ")

    manifest = _read_json(MANIFEST_PATH)
    benchmark = _read_json(BENCHMARK_PATH)
    _validate_inputs(manifest, benchmark)

    selected_pages = manifest["selected_pages"]
    page_numbers = [item["pdf_page"] for item in selected_pages]
    metadata_by_page = {
        item["pdf_page"]: {"chapter": item["chapter"]} for item in selected_pages
    }
    pdf_path = ROOT / manifest["local_path"]
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Corpus PDF not found: {pdf_path}")

    pages = load_pdf(
        pdf_path,
        document_id=manifest["source_id"],
        source=manifest["source_path_or_url"],
        page_numbers=page_numbers,
        metadata_by_page=metadata_by_page,
    )
    if len(pages) != manifest["selected_page_count"]:
        raise ValueError("Extracted page count does not match the manifest")
    for page in pages:
        if not page.text:
            raise ValueError(f"Selected PDF page {page.page} has no extractable text")
        expected_chars = next(
            item["normalized_characters"]
            for item in selected_pages
            if item["pdf_page"] == page.page
        )
        if _normalized_character_count(page.text) != expected_chars:
            raise ValueError(
                f"Normalized text length changed for PDF page {page.page}; "
                "review the manifest before running."
            )

    page_text = {page.page: page.text for page in pages}
    for question in benchmark["records"]:
        evidence = _normalize_for_match(question["evidence_quote"])
        if evidence not in _normalize_for_match(page_text[question["gold_page"]]):
            raise ValueError(f"{question['id']} evidence does not match its gold page")

    chunking_started = time.perf_counter()
    chunks_a = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=500, overlap=50)
    ]
    chunking_seconds_a = time.perf_counter() - chunking_started
    chunking_started = time.perf_counter()
    chunks_b = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=500)
    ]
    chunking_seconds_b = time.perf_counter() - chunking_started
    chunking_seconds = {"A": chunking_seconds_a, "B": chunking_seconds_b, "C": 0.0}
    chunks_by_strategy = {"A": chunks_a, "B": chunks_b, "C": chunks_b}

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing; configure starter/.env")
    embedder = FastEmbedder(model_name=os.getenv("EMBEDDING_MODEL") or MODEL_NAME)
    store = PgVectorStore(database_url, dimensions=embedder.dimensions)
    store.create_schema()

    embedding_tokens: dict[str, int | None] = {}
    embedding_seconds: dict[str, float] = {}
    started = time.perf_counter()
    vectors_a = embedder.embed_documents([chunk.text for chunk in chunks_a])
    embedding_seconds["A"] = time.perf_counter() - started
    embedding_tokens["A"] = None

    started = time.perf_counter()
    vectors_b = embedder.embed_documents([chunk.text for chunk in chunks_b])
    embedding_seconds["B"] = time.perf_counter() - started
    embedding_tokens["B"] = None
    embedding_seconds["C"] = 0.0
    embedding_tokens["C"] = None

    started = time.perf_counter()
    query_vectors = embedder.embed_documents(
        [question["question"] for question in benchmark["records"]]
    )
    query_embedding_seconds = time.perf_counter() - started
    query_embedding_tokens = None

    vectors_by_strategy = {"A": vectors_a, "B": vectors_b, "C": vectors_b}
    store_seconds: dict[str, float] = {}
    for strategy in ("A", "B", "C"):
        started = time.perf_counter()
        store.upsert(strategy, chunks_by_strategy[strategy], vectors_by_strategy[strategy])
        store_seconds[strategy] = time.perf_counter() - started

    traces: list[dict[str, Any]] = []
    search_seconds: dict[str, float] = {}
    for strategy in ("A", "B", "C"):
        started = time.perf_counter()
        for question, query_vector in zip(benchmark["records"], query_vectors):
            filters = question.get("filter") if strategy == "C" else None
            hits = store.search(strategy, query_vector, top_k=3, filters=filters)
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            traces.append(trace)
        search_seconds[strategy] = time.perf_counter() - started

    RESULTS_PATH.mkdir(exist_ok=True)
    output_dir = RESULTS_PATH

    summary = summarize(traces)
    _write_json(output_dir / "metrics.json", summary)
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for trace in traces:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")

    strategy_stats: dict[str, Any] = {}
    for strategy, chunks in chunks_by_strategy.items():
        strategy_stats[strategy] = {
            **_chunk_stats(chunks),
            "chunking_seconds": chunking_seconds[strategy],
            "document_embedding_seconds": embedding_seconds[strategy],
            "document_embedding_tokens": embedding_tokens[strategy],
            "store_seconds": store_seconds[strategy],
            "search_seconds": search_seconds[strategy],
            "document_vectors_reused_from": "B" if strategy == "C" else None,
        }

    run_metadata = {
        "run_id": run_id,
        "started_at_utc": run_started_at.isoformat(),
        "source_id": manifest["source_id"],
        "selected_pdf_pages": page_numbers,
        "question_count": len(benchmark["records"]),
        "embedding": {
            "provider": "FastEmbed local CPU",
            "requested_model": embedder.model_name,
            "dimensions": embedder.dimensions,
            "document_tokens_by_strategy": embedding_tokens,
            "query_tokens_shared": query_embedding_tokens,
            "total_input_tokens": None,
            "token_usage_available": False,
            "query_embedding_seconds": query_embedding_seconds,
            "estimated_cost_usd": 0.0,
            "cost_note": "Local inference; no embedding API token charges.",
        },
        "vector_store": {
            "backend": "PostgreSQL + pgvector",
            "metric": "cosine distance",
            "search": "exact",
        },
        "strategies": strategy_stats,
    }
    sample_pages = _write_audit_sheets(output_dir, traces, chunks_by_strategy, pages)
    run_metadata["coherence_sample_pages"] = sample_pages
    run_metadata["output_directory"] = str(output_dir.relative_to(ROOT))
    _write_json(output_dir / "run_metadata.json", run_metadata)
    logger.info("Benchmark artifacts saved to %s", output_dir)


if __name__ == "__main__":
    main()
