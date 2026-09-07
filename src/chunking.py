from pydantic import BaseModel, Field
from typing import ClassVar
from pathlib import Path
import ast
import re


class Chunker(BaseModel):
    def line_start(self, text: str) -> list[int]:
        start = [0]
        for line in text.splitlines(keepends=True):
            start.append(start[-1] + len(line))
        return start

    def real_start_line(self, node) -> int:
        if node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno

    def _hard_cut(
        self, text: str, start: int, end: int, max_chunk: int
    ) -> list[tuple[int, int, str]]:
        """Last-resort fallback: plain character windows of max_chunk_size."""
        out = []
        for i in range(start, end, max_chunk):
            j = min(i + max_chunk, end)
            out.append((i, j, text[i:j]))
        return out

    def _python_chunker(self, text: str, max_chunk: int) -> list[tuple[int, int, str]]:
        """
        returns (start, end, chunk)
        """
        try:
            tree = ast.parse(text)
        except (SyntaxError, IndentationError) as e:
            print(e)
            return []

        chunks = []
        starts = self.line_start(text)
        lines = text.splitlines()
        def_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

        for node in tree.body:
            if isinstance(node, def_types):
                start_char = starts[self.real_start_line(node) - 1]
                end_char = starts[node.end_lineno]

                if end_char - start_char <= max_chunk:
                    chunks.append((start_char, end_char, text[start_char:end_char]))

                elif isinstance(node, ast.ClassDef):
                    header = lines[node.lineno - 1]
                    # -1 for the "\n" between header and method
                    budget = max_chunk - len(header) - 1
                    method_types = (ast.FunctionDef, ast.AsyncFunctionDef)
                    for child in node.body:
                        if isinstance(child, method_types):
                            m_start = starts[self.real_start_line(child) - 1]
                            m_end = starts[child.end_lineno]
                            if m_end - m_start <= budget:
                                chunk_text = header + "\n" + text[m_start:m_end]
                                chunks.append((m_start, m_end, chunk_text))
                            else:
                                chunks.extend(
                                    self._hard_cut(text, m_start, m_end, max_chunk)
                                )

                else:  # a huge plain function: never drop content
                    chunks.extend(self._hard_cut(text, start_char, end_char, max_chunk))
                continue

            start_char = starts[node.lineno - 1]
            end_char = starts[node.end_lineno]
            chunk_text = text[start_char:end_char]
            if not chunk_text.strip():
                continue
            if end_char - start_char <= max_chunk:
                chunks.append((start_char, end_char, chunk_text))
            else:
                chunks.extend(self._hard_cut(text, start_char, end_char, max_chunk))

        return chunks

    def _text_md_chunker(self, text: str, max_chunk: int) -> list[tuple[int, int, str]]:
        """
        returns (start, end, chunk)
        """
        cut_points = [0]
        for m in re.finditer(r"^#{1,6}\s", text, flags=re.MULTILINE):
            if m.start() != 0:
                cut_points.append(m.start())
        cut_points.append(len(text))

        chunks = []
        for a, b in zip(cut_points, cut_points[1:]):
            if not text[a:b].strip():
                continue
            if b - a <= max_chunk:
                chunks.append((a, b, text[a:b]))
            else:
                chunks.extend(self._hard_cut(text, a, b, max_chunk))
        return chunks


class DatasetChunker(Chunker):
    max_chunk: int = Field(ge=0, le=2000)

    EXT_DISPATCH: ClassVar[dict[str, str]] = {
        ".py": "_python_chunker",
        ".md": "_text_md_chunker",
        ".markdown": "_text_md_chunker",
    }

    def chunk_file(self, path: Path) -> list[dict]:
        method_name = self.EXT_DISPATCH.get(path.suffix)
        if method_name is None:
            return []

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
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

    def chunk_directory(self, path: Path):
        from tqdm import tqdm
        dataset = []

        for thing in tqdm(sorted(path.rglob("*")),
                          unit=" files", desc="Chunking: "):
            if thing.is_file():
                dataset.extend(self.chunk_file(thing))

        return dataset
