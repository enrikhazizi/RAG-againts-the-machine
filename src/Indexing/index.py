from pathlib import Path


class DatasetChunker(Chunker):
    EXT_DISPATCH = {
        ".py": "python_chunker",
        ".md": "text_md_chunker",
        ".markdown": "text_md_chunker",
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
        raw_chunks = method(text)

        return [
            {
                "text": chunk,
                "source": str(path),
                "chunk_index": i,
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    def chunk_directory