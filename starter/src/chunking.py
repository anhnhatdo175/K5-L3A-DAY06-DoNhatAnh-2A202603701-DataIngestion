"""Fixed-size and paragraph-aware chunking with stable source offsets."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    strategy: str,
    index: int,
    start: int,
    end: int,
) -> Chunk:
    text = page.text[start:end].strip()
    if not text:
        raise ValueError("Cannot create an empty chunk")
    identity = f"{strategy}\0{page.document_id}\0{page.page}\0{index}\0{start}\0{text}"
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    metadata = dict(page.metadata)
    metadata.update(
        {
            "chunk_method": strategy,
            "chunk_index": index,
            "char_start": start,
            "char_end": end,
        }
    )
    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split one page into overlapping character windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            chunks.append(_make_chunk(page, "fixed", len(chunks), start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def _split_oversized(
    text: str,
    absolute_start: int,
    max_size: int,
) -> list[tuple[int, int]]:
    if len(text) <= max_size:
        return [(absolute_start, absolute_start + len(text))]

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence:
            continue
        local_start = text.find(sentence, cursor)
        local_end = local_start + len(sentence)
        cursor = local_end
        if len(sentence) <= max_size:
            ranges.append((absolute_start + local_start, absolute_start + local_end))
            continue

        words = list(re.finditer(r"\S+", sentence))
        word_index = 0
        while word_index < len(words):
            first = words[word_index]
            start = first.start()
            end = first.end()
            if end - start > max_size:
                ranges.extend(
                    (
                        absolute_start + local_start + offset,
                        absolute_start + local_start + min(offset + max_size, end),
                    )
                    for offset in range(start, end, max_size)
                )
                word_index += 1
                continue
            word_index += 1
            while word_index < len(words):
                next_end = words[word_index].end()
                if next_end - start > max_size:
                    break
                end = next_end
                word_index += 1
            ranges.append((absolute_start + local_start + start, absolute_start + local_start + end))
    return ranges


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack whole paragraphs together, splitting only when a paragraph is too long."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    if not page.text:
        return []

    paragraph_ranges: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n", page.text):
        block = page.text[start : separator.start()]
        leading = len(block) - len(block.lstrip())
        trailing = len(block.rstrip())
        if trailing > leading:
            paragraph_ranges.append((start + leading, start + trailing))
        start = separator.end()
    block = page.text[start:]
    leading = len(block) - len(block.lstrip())
    trailing = len(block.rstrip())
    if trailing > leading:
        paragraph_ranges.append((start + leading, start + trailing))

    units: list[tuple[int, int]] = []
    for paragraph_start, paragraph_end in paragraph_ranges:
        paragraph = page.text[paragraph_start:paragraph_end]
        units.extend(_split_oversized(paragraph, paragraph_start, max_size))
    if not units:
        return []

    packed_ranges: list[tuple[int, int]] = []
    current_start, current_end = units[0]
    for next_start, next_end in units[1:]:
        if next_end - current_start <= max_size:
            current_end = next_end
        else:
            packed_ranges.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    packed_ranges.append((current_start, current_end))

    return [
        _make_chunk(page, "paragraph", index, chunk_start, chunk_end)
        for index, (chunk_start, chunk_end) in enumerate(packed_ranges)
    ]
