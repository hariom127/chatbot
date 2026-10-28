import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split on paragraph/sentence/space boundaries where possible, with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            # prefer a natural break in the second half of the window
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut >= chunk_size // 2:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc: dict, chunk_size: int, overlap: int) -> list[Chunk]:
    """doc keys: id, title, url, modified_time, text. Returns [] for empty docs."""
    text = clean_text(doc.get("text", ""))
    pieces = split_text(text, chunk_size, overlap)
    return [
        Chunk(
            id=f"{doc['id']}:{i}",
            text=piece,
            metadata={
                "doc_id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "chunk_index": i,
                "modified_time": doc["modified_time"],
            },
        )
        for i, piece in enumerate(pieces)
    ]
