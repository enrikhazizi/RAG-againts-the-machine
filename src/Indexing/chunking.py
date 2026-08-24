import re
from pydantic import BaseModel, Field


class chunker(BaseModel):

    def _python_chunker(self, text: str) -> list[str]:
        pieces = re.split(r"(?=\bdef\b|\bclass\b)", text.strip())
        chunks = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if self._rough_token_count(piece) <= self.max_tokens:
                chunks.append(piece)
        return chunks

    def _text_md_chunker(self, text):
        pieces = re.split(r"(?=^#{1,6}\s)", text.strip(), flags=re.MULTILINE)
        return [p.strip() for p in pieces if p.strip()]
