import unittest
from pathlib import Path

from src.chunking import by_paragraph, fixed_size
from src.embeddings import FastEmbedder
from src.evaluation import evaluate_query, summarize
from src.ingestion import load_pdf, normalize_text
from src.models import DocumentPage
from src.run import _read_json, _validate_inputs
from src.store import PgVectorStore


ROOT = Path(__file__).resolve().parents[2]


class FakeTextEmbedding:
    def __init__(self, vector_size=384):
        self.vector_size = vector_size
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return iter(
            [[float(index)] * self.vector_size for index in range(len(texts))]
        )


class PipelineTests(unittest.TestCase):
    def test_normalize_text_preserves_paragraphs_and_normalizes_whitespace(self):
        text = "  A   \r\nB\r\n\r\n\r\nC\u00a0 D  "
        self.assertEqual(normalize_text(text), "A B\n\nC D")

    def test_load_pdf_extracts_selected_physical_pages_and_metadata(self):
        pdf_path = ROOT / "data" / "introduction-sociology-3e_-_WEB.pdf"
        pages = load_pdf(
            pdf_path,
            document_id="sociology",
            source="book.pdf",
            page_numbers=[77, 107],
            metadata_by_page={77: {"chapter": 3}, 107: {"chapter": 4}},
        )
        self.assertEqual([page.page for page in pages], [77, 107])
        self.assertTrue(all(page.text for page in pages))
        self.assertEqual([page.metadata["chapter"] for page in pages], [3, 4])

    def test_load_pdf_rejects_duplicate_and_invalid_page_numbers(self):
        pdf_path = ROOT / "data" / "introduction-sociology-3e_-_WEB.pdf"
        with self.assertRaises(ValueError):
            load_pdf(pdf_path, "id", "book.pdf", [77, 77])
        with self.assertRaises(TypeError):
            load_pdf(pdf_path, "id", "book.pdf", [True])

    def test_fixed_size_chunks_keep_offsets_and_overlap(self):
        page = DocumentPage("book", "book.pdf", 1, "0123456789abcdefghij")
        chunks = fixed_size(page, size=10, overlap=2)
        self.assertEqual([chunk.text for chunk in chunks], [
            "0123456789",
            "89abcdefgh",
            "ghij",
        ])
        self.assertEqual(chunks[0].text[-2:], chunks[1].text[:2])
        self.assertTrue(
            all(
                page.text[chunk.metadata["char_start"] : chunk.metadata["char_end"]].strip()
                == chunk.text
                for chunk in chunks
            )
        )

    def test_paragraph_chunking_respects_limit_and_preserves_text(self):
        text = "Intro sentence.\n\n" + ("oversized " * 12) + "\n\nEnding."
        page = DocumentPage("book", "book.pdf", 1, text)
        chunks = by_paragraph(page, max_size=32)
        reconstructed = " ".join(chunk.text for chunk in chunks)
        self.assertTrue(all(len(chunk.text) <= 32 for chunk in chunks))
        self.assertEqual(" ".join(reconstructed.split()), " ".join(text.split()))
        self.assertTrue(
            all(
                page.text[chunk.metadata["char_start"] : chunk.metadata["char_end"]].strip()
                == chunk.text
                for chunk in chunks
            )
        )

    def test_chunking_rejects_invalid_sizes(self):
        page = DocumentPage("book", "book.pdf", 1, "text")
        with self.assertRaises(ValueError):
            fixed_size(page, size=0, overlap=0)
        with self.assertRaises(ValueError):
            fixed_size(page, size=5, overlap=5)
        with self.assertRaises(ValueError):
            by_paragraph(page, max_size=0)

    def test_local_embedder_batches_and_preserves_input_order(self):
        model = FakeTextEmbedding()
        embedder = FastEmbedder(model=model, batch_size=2)
        vectors = embedder.embed_documents(["first", "second", "third"])
        self.assertEqual(model.calls, [["first", "second"], ["third"]])
        self.assertEqual([vector[0] for vector in vectors], [0.0, 1.0, 0.0])
        self.assertEqual(len(vectors[0]), 384)
        self.assertEqual(embedder.embed_query("query")[0], 0.0)
        self.assertEqual(model.calls[-1], ["query"])

    def test_local_embedder_rejects_empty_inputs_and_bad_dimensions(self):
        embedder = FastEmbedder(model=FakeTextEmbedding(vector_size=3))
        with self.assertRaises(ValueError):
            embedder.embed_documents(["   "])
        with self.assertRaises(ValueError):
            embedder.embed_documents(["text"])
        with self.assertRaises(ValueError):
            embedder.embed_query("   ")

    def test_evaluation_requires_gold_page_and_evidence(self):
        question = {
            "id": "Q1",
            "split": "test",
            "question": "What is culture?",
            "expected_answer": "Shared practices",
            "gold_source_id": "book",
            "gold_page": 77,
            "evidence_quote": "shared practices and beliefs",
        }
        good = {
            "document_id": "book",
            "page": 77,
            "text": "Culture includes shared practices and beliefs.",
        }
        wrong_page = {**good, "page": 78}
        no_evidence = {**good, "text": "Culture includes values."}
        trace = evaluate_query(question, [good, wrong_page, no_evidence])
        self.assertTrue(trace["hit_at_1"])
        self.assertTrue(trace["evidence_hit_at_3"])
        self.assertFalse(trace["retrieved"][1]["evidence_hit"])
        self.assertEqual(
            summarize([{"strategy": "A", **trace}])["A"]["test"]["questions"],
            1,
        )

    def test_store_validates_vector_shape_and_finite_values(self):
        store = PgVectorStore("postgresql://unused", dimensions=2)
        store._validate_vector([0.1, 0.2])
        with self.assertRaises(ValueError):
            store._validate_vector([0.1])
        with self.assertRaises(ValueError):
            store._validate_vector([0.1, float("inf")])

    def test_checked_in_manifest_and_benchmark_satisfy_pipeline_contract(self):
        manifest = _read_json(ROOT / "data" / "manifest.json")
        benchmark = _read_json(ROOT / "benchmark" / "questions.json")
        _validate_inputs(manifest, benchmark)

    def test_input_validation_rejects_malformed_filter(self):
        manifest = _read_json(ROOT / "data" / "manifest.json")
        benchmark = _read_json(ROOT / "benchmark" / "questions.json")
        benchmark["records"][0]["filter"] = ["chapter"]
        with self.assertRaises(ValueError):
            _validate_inputs(manifest, benchmark)


if __name__ == "__main__":
    unittest.main()
