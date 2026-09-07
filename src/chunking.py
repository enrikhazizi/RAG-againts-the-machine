"""AST-aware Python chunking and heading-based Markdown chunking."""
from pydantic import BaseModel, Field
from typing import ClassVar, TypedDict
from pathlib import Path
from tqdm import tqdm
import ast
import re


class ChunkRecord(TypedDict):
    text: str
    source: str
    chunk_index: int
    start: int
    end: int


ChunkSpan = tuple[int, int, str]
DefNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class Chunker(BaseModel):
    def line_start(self, text: str) -> list[int]:
        start = [0]
        for line in text.splitlines(keepends=True):
            start.append(start[-1] + len(line))
        return start

    def real_start_line(self, node: DefNode) -> int:
        if node.decorator_list:
            lineno = node.decorator_list[0].lineno
            return lineno if isinstance(lineno, int) else 1
        lineno = node.lineno
        return lineno if isinstance(lineno, int) else 1

    def _end_line(self, node: ast.AST, n_starts: int) -> int:
        end = getattr(node, "end_lineno", None)
        if not isinstance(end, int) or end >= n_starts:
            return n_starts - 1
        return end

    def _hard_cut(
        self, text: str, start: int, end: int, max_chunk: int
    ) -> list[ChunkSpan]:
        if max_chunk <= 0 or start >= end:
            return []
        out: list[ChunkSpan] = []
        for i in range(start, end, max_chunk):
            j = min(i + max_chunk, end)
            out.append((i, j, text[i:j]))
        return out

    def _python_chunker(
        self, text: str, max_chunk: int
    ) -> list[ChunkSpan]:
        try:
            tree = ast.parse(text)
        except (SyntaxError, IndentationError) as exc:
            print(exc)
            return []

        chunks: list[ChunkSpan] = []
        starts = self.line_start(text)
        lines = text.splitlines()
        def_types = (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )

        for node in tree.body:
            if isinstance(node, def_types):
                start_char = starts[self.real_start_line(node) - 1]
                end_char = starts[self._end_line(node, len(starts))]

                if end_char - start_char <= max_chunk:
                    chunk = text[start_char:end_char]
                    chunks.append((start_char, end_char, chunk))

                elif isinstance(node, ast.ClassDef):
                    header = lines[node.lineno - 1]
                    # -1 for the "\n" between header and method
                    budget = max_chunk - len(header) - 1
                    method_types = (
                        ast.FunctionDef, ast.AsyncFunctionDef,
                    )
                    for child in node.body:
                        if isinstance(child, method_types):
                            m_start = starts[
                                self.real_start_line(child) - 1
                            ]
                            m_end = starts[
                                self._end_line(child, len(starts))
                            ]
                            if m_end - m_start <= budget:
                                chunk_text = (
                                    header + "\n" + text[m_start:m_end]
                                )
                                chunks.append(
                                    (m_start, m_end, chunk_text)
                                )
                            else:
                                chunks.extend(
                                    self._hard_cut(
                                        text, m_start, m_end, max_chunk
                                    )
                                )

                else:
                    # a huge plain function: never drop content
                    chunks.extend(
                        self._hard_cut(
                            text, start_char, end_char, max_chunk
                        )
                    )
                continue

            lineno = node.lineno if isinstance(node.lineno, int) else 1
            start_char = starts[lineno - 1]
            end_char = starts[self._end_line(node, len(starts))]
            chunk_text = text[start_char:end_char]
            if not chunk_text.strip():
                continue
            if end_char - start_char <= max_chunk:
                chunks.append((start_char, end_char, chunk_text))
            else:
                chunks.extend(
                    self._hard_cut(
                        text, start_char, end_char, max_chunk
                    )
                )

        return chunks

    def _text_md_chunker(
        self, text: str, max_chunk: int
    ) -> list[ChunkSpan]:
        cut_points = [0]
        for match in re.finditer(r"^#{1,6}\s", text, flags=re.MULTILINE):
            if match.start() != 0:
                cut_points.append(match.start())
        cut_points.append(len(text))

        chunks: list[ChunkSpan] = []
        for start, end in zip(cut_points, cut_points[1:]):
            if not text[start:end].strip():
                continue
            if end - start <= max_chunk:
                chunks.append((start, end, text[start:end]))
            else:
                chunks.extend(
                    self._hard_cut(text, start, end, max_chunk)
                )
        return chunks


class DatasetChunker(Chunker):
    max_chunk: int = Field(ge=0, le=2000)

    EXT_DISPATCH: ClassVar[dict[str, str]] = {
        ".py": "_python_chunker",
        ".md": "_text_md_chunker",
        ".markdown": "_text_md_chunker",
        ".txt": "_text_md_chunker",
    }

    def chunk_file(self, path: Path) -> list[ChunkRecord]:
        """Chunk a single file according to its suffix.

        Args:
            path: File to split.

        Returns:
            Chunk records, or an empty list for unsupported / unreadable
            files.
        """
        method_name = self.EXT_DISPATCH.get(path.suffix)
        if method_name is None:
            return []

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return []

        method = getattr(self, method_name)
        raw_chunks = method(text, self.max_chunk)

        return [
            {
                "text": chunk_text,
                "source": str(path.as_posix()),
                "chunk_index": i,
                "start": start,
                "end": end,
            }
            for i, (start, end, chunk_text) in enumerate(raw_chunks)
        ]

    def chunk_directory(self, path: Path) -> list[ChunkRecord]:
        dataset: list[ChunkRecord] = []
        if not path.exists():
            print(f"Corpus not found: {path}")
            return dataset

        for thing in tqdm(
            sorted(path.rglob("*")),
            unit=" files",
            desc="Chunking: ",
        ):
            if thing.is_file():
                dataset.extend(self.chunk_file(thing))

        return dataset
