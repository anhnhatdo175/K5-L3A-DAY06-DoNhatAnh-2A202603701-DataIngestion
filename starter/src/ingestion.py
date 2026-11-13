"""Read PDF pages while preserving physical-page and source metadata."""
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    page_numbers: Sequence[int] | None = None,
    metadata_by_page: Mapping[int, Mapping[str, object]] | None = None,
) -> list[DocumentPage]:
    """Extract selected 1-based physical PDF pages and retain their metadata."""
    reader = PdfReader(str(path))
    selected_pages = list(page_numbers) if page_numbers is not None else list(
        range(1, len(reader.pages) + 1)
    )
    if len(selected_pages) != len(set(selected_pages)):
        raise ValueError("page_numbers must not contain duplicates")
    if any(not isinstance(page, int) or isinstance(page, bool) for page in selected_pages):
        raise TypeError("PDF page numbers must be integers")
    if any(page < 1 or page > len(reader.pages) for page in selected_pages):
        raise ValueError(f"PDF page numbers must be between 1 and {len(reader.pages)}")

    page_metadata = metadata_by_page or {}
    documents: list[DocumentPage] = []
    for physical_page in selected_pages:
        raw_text = reader.pages[physical_page - 1].extract_text() or ""
        if "\ufffd" in raw_text:
            logger.warning("PDF page %s contains replacement characters", physical_page)
        text = normalize_text(raw_text)
        if not text:
            logger.warning("PDF page %s has no extractable text", physical_page)
        documents.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=physical_page,
                text=text,
                metadata=dict(page_metadata.get(physical_page, {})),
            )
        )
    return documents


def normalize_text(text: str) -> str:
    """Normalize Unicode and line wrapping while preserving paragraph breaks."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", normalized)

    paragraphs: list[str] = []
    for block in re.split(r"\n[ \t]*\n", normalized):
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in block.splitlines()]
        paragraph = " ".join(line for line in lines if line)
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)
