"""Load Markdown/text documents and split them into retrievable chunks.

Chunks follow the document's own structure: each heading starts a new section,
paragraphs are packed into chunks of up to ``max_words``, and the section heading
is prepended to every chunk so a passage still makes sense out of context.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    id: int
    source: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} § {self.heading}" if self.heading else self.source


def iter_documents(paths: Iterable[str | Path]) -> Iterator[tuple[str, str]]:
    """Yield (source name, text) for every supported file under the given paths."""
    for path in map(Path, paths):
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        else:
            raise FileNotFoundError(f"corpus path not found: {path}")
        for file in files:
            name = file.name if path.is_file() else str(file.relative_to(path))
            yield name, file.read_text(encoding="utf-8")


def _sections(text: str) -> Iterator[tuple[str, str]]:
    heading, lines = "", []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            if any(s.strip() for s in lines):
                yield heading, "\n".join(lines)
            heading, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    if any(s.strip() for s in lines):
        yield heading, "\n".join(lines)


def _split_long(words: list[str], max_words: int, overlap: int) -> Iterator[str]:
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        yield " ".join(words[start : start + max_words])
        if start + max_words >= len(words):
            break


def chunk_text(
    source: str, text: str, max_words: int = 160, overlap_words: int = 30, start_id: int = 0
) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(heading: str, body: str) -> None:
        body = body.strip()
        if body:
            prefix = f"{heading}\n" if heading else ""
            chunks.append(Chunk(start_id + len(chunks), source, heading, prefix + body))

    for heading, section in _sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        buffer: list[str] = []
        size = 0
        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) > max_words:
                emit(heading, "\n\n".join(buffer))
                buffer, size = [], 0
                for piece in _split_long(words, max_words, overlap_words):
                    emit(heading, piece)
                continue
            if size + len(words) > max_words and buffer:
                emit(heading, "\n\n".join(buffer))
                buffer, size = [], 0
            buffer.append(paragraph)
            size += len(words)
        emit(heading, "\n\n".join(buffer))
    return chunks


def load_chunks(paths: Iterable[str | Path], max_words: int = 160) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in iter_documents(paths):
        chunks.extend(chunk_text(source, text, max_words=max_words, start_id=len(chunks)))
    if not chunks:
        raise ValueError("corpus is empty: no .md or .txt content found")
    return chunks


def default_corpus_paths() -> list[Path]:
    """The project's own documentation: packaged into the wheel, or read from the repo."""
    packaged = Path(__file__).parent / "bundled_docs"
    if packaged.is_dir():
        return [packaged]
    repo = Path(__file__).resolve().parents[3]
    paths = [p for p in (repo / "docs", repo / "README.md") if p.exists()]
    if not paths:
        raise FileNotFoundError("no default corpus found; set INFERSCALE_RAG_CORPUS")
    return paths
