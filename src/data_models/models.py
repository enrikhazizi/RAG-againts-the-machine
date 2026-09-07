from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class FullSource(MinimalSource):
    text: str


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class FullSearchResults(MinimalSearchResults):
    retrieved_sources: List[FullSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


    "bm25s>=0.3.10",
    "fire>=0.7.1",
    "nltk>=3.10.3",
    "numpy>=2.4.6",
    "pydantic>=2.13.4",
    "pystemmer>=3.1.0",
    "vllm>=0.27.1; sys_platform == 'linux'",