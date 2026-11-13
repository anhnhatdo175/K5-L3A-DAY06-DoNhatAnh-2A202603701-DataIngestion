"""Persistent exact vector search using PostgreSQL and pgvector."""
import math
from collections.abc import Mapping, Sequence
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be empty")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the vector table and metadata indexes if they do not exist."""
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS ingestion_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )
            embedding_type = connection.execute(
                """
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'ingestion_chunks'::regclass
                  AND attname = 'embedding'
                  AND NOT attisdropped
                """
            ).fetchone()
            expected_type = f"vector({self.dimensions})"
            if embedding_type and embedding_type[0] != expected_type:
                row_count = connection.execute(
                    "SELECT COUNT(*) FROM ingestion_chunks"
                ).fetchone()[0]
                if row_count:
                    raise RuntimeError(
                        f"Existing ingestion_chunks uses {embedding_type[0]} and contains "
                        f"{row_count} rows; refusing to change it to {expected_type}."
                    )
                connection.execute(
                    "ALTER TABLE ingestion_chunks "
                    f"ALTER COLUMN embedding TYPE {expected_type} "
                    f"USING embedding::{expected_type}"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ingestion_chunks_strategy_idx "
                "ON ingestion_chunks (strategy)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ingestion_chunks_metadata_idx "
                "ON ingestion_chunks USING GIN (metadata)"
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace this strategy's rows for the included documents transactionally."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            raise ValueError("Refusing to replace a strategy with no chunks")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk IDs must be unique within a strategy")
        for vector in vectors:
            self._validate_vector(vector)

        document_ids = sorted({chunk.document_id for chunk in chunks})
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                "DELETE FROM ingestion_chunks "
                "WHERE strategy = %s AND document_id = ANY(%s)",
                (strategy, document_ids),
            )
            for chunk, vector in zip(chunks, vectors):
                connection.execute(
                    """
                    INSERT INTO ingestion_chunks
                        (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        strategy,
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.text,
                        Jsonb(chunk.metadata),
                        self._vector_literal(vector),
                    ),
                )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run exact cosine search, optionally constrained by supported metadata."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._validate_vector(query_vector)
        if filters:
            unsupported = set(filters) - {"chapter"}
            if unsupported:
                raise ValueError(f"Unsupported metadata filters: {sorted(unsupported)}")
            chapter = filters.get("chapter")
            if type(chapter) is not int or chapter not in (3, 4):
                raise ValueError("chapter filter must be integer 3 or 4")

        query = """
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM ingestion_chunks
            WHERE strategy = %s
        """
        params: list[Any] = [self._vector_literal(query_vector), strategy]
        if filters:
            query += " AND metadata @> %s"
            params.append(Jsonb(dict(filters)))
        query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend((self._vector_literal(query_vector), top_k))

        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(vector)}")
        if not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("Vector values must be finite numbers")

    @staticmethod
    def _vector_literal(vector: Sequence[float]) -> str:
        return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"
