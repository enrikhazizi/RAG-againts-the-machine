import re
from pydantic import BaseModel, Field


class chunker(BaseModel):
    max_tokens: int = Field(le=2000, ge=1, default=2000)

    def python_chunker(self, text):

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_tokens + sentence_tokens <= self.max_tokens:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def text_md_chunker(self, text):
        pass