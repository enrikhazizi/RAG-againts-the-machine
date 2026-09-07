from data_models import (FullSource, UnansweredQuestion,
                         RagDataset, StudentSearchResults,
                         MinimalSource, MinimalSearchResults,
                         StudentSearchResultsAndAnswer, MinimalAnswer,
                         )
from chunking import DatasetChunker
from bm25s.tokenization import Tokenizer
from ollama import generate
from pathlib import Path
from tqdm import tqdm
from typing import List
import Stemmer
import json
import re
import fire
import bm25s

INDEX_DIR = Path("data/processed/vllm")
QUERY_DIR = Path("data/dataset/UnansweredQuestions")
SEARCH_OUT_DIR = Path("data/output/search_results")
SAVE_ANS = Path("data/output/search_result_and_answer")
MODEL = "Qwen3:0.6b"


def _tokenizer() -> Tokenizer:
    return Tokenizer(stemmer=Stemmer.Stemmer("english"), stopwords=[], splitter=code_split)


def code_split(text: str) -> list[str]:
    """Split text into lowercase word tokens, cutting identifiers apart."""
    tokens: list[str] = []
    # Step 1: cut on anything that is not a letter/digit/underscore
    for word in re.split(r"[^0-9a-zA-Z_]+", text):
        if not word:
            continue
        word_lower = word.lower()
        tokens.append(word_lower)
        if "_" in word_lower:
            snake_pieces = [p for p in word_lower.split("_") if p]
            tokens.extend(snake_pieces)
        # TODO (bonus): split camelCase pieces too
    return tokens


def _display_path(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return path


def _as_question(query: UnansweredQuestion | str) -> UnansweredQuestion:
    if isinstance(query, UnansweredQuestion):
        return query
    return UnansweredQuestion(question=query)


def _load_json(path: Path) -> RagDataset:
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    questions = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        questions.extend(
            UnansweredQuestion(
                question_id=data["question_id"],
                question=data["question"],
            )
            for data in json_file["rag_questions"]
        )
    return RagDataset(rag_questions=questions)


class Rag():

    def index(self, max_chunk_size=2000):
        chunker = DatasetChunker(max_chunk=max_chunk_size)
        my_dir = Path("data/raw/vllm-0.10.1").resolve()

        corpus = chunker.chunk_directory(my_dir)

        tokenizer = _tokenizer()

        all_ids = []
        for chunk in tqdm(corpus, desc="Tokenizing chunks", unit=" chunk"):
            path_words = Path(chunk["source"]).stem.replace("-", " ")
            tokens = tokenizer.tokenize(
                [path_words + " " + chunk["text"]], update_vocab=True,
                return_as="ids", show_progress=False
            )
            all_ids.extend(tokens)

        corpus_tokens = tokenizer.to_tokenized_tuple(all_ids)

        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(INDEX_DIR)
        print(
            f"Ingestion complete! Indexed {len(corpus)} "
            "chunks under data/processed/"
        )

    def _retrieve(
        self,
        query: UnansweredQuestion,
        k: int = 5,
        path: Path = INDEX_DIR,
        retriever=None,
    ) -> list[FullSource]:
        if retriever is None:
            retriever = bm25s.BM25.load(path, load_corpus=True)

        query_tokens = _tokenizer().tokenize(
            [query.question],
            update_vocab=True,
            return_as="string",
            show_progress=False,
        )
        docs, _scores = retriever.retrieve(query_tokens, k=k)

        retrieved: list[FullSource] = []
        for doc in docs[0]:
            retrieved.append(FullSource(
                file_path=_display_path(doc["source"]),
                first_character_index=doc["start"],
                last_character_index=doc["end"],
                text=doc["text"],
            ))
        return retrieved

    def search(self, query: str, k: int = 5, path: Path = INDEX_DIR) -> None:
        question = _as_question(query)
        for source in self._retrieve(question, k=k, path=path):
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: Path,
        k: int = 5,
        path: Path = INDEX_DIR,
        save_directory: Path = SEARCH_OUT_DIR
    ) -> List[MinimalSearchResults] | None:
        dataset = _load_json(Path(dataset_path))
        retriever = bm25s.BM25.load(path, load_corpus=True)
        results: list[MinimalSearchResults] = []

        for question in tqdm(dataset.rag_questions, desc="Searching"):
            retrieved = self._retrieve(question, k=k, retriever=retriever)
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=src.file_path,
                        first_character_index=src.first_character_index,
                        last_character_index=src.last_character_index,
                    )
                    for src in retrieved
                ],
            ))

        output = StudentSearchResults(search_results=results, k=k)
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        out_file = save_directory / Path(dataset_path).name
        out_file.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {out_file}")

    def _sources_text(self, sources: List[MinimalSource]) -> str:
        parts = []
        for src in sources:
            text = Path(src.file_path).read_text(encoding="utf-8", errors="replace")
            chunk = text[src.first_character_index:src.last_character_index]
            parts.append(
                f"{src.file_path} "
                f"[{src.first_character_index}:{src.last_character_index}]\n"
                f"{chunk}"
            )
        return "\n\n".join(parts)

    def _prompt(self, question: str, context: str) -> str:
        if not context:
            context = "(no chunks retrieved — index may be missing)"
        return (
            "You are answering questions about the vLLM codebase "
            "from retrieved source chunks.\n"
            "Rules:\n"
            "- Use only the chunks below. Do not "
            "invent files, functions, or behavior.\n"
            "- If the chunks do not contain enough evidence, say so "
            "and answer only what they support.\n"
            "- Cite every claim with file path and character range [start:end].\n"
            f"<question>\n{question}\n</question>\n"
            f"<retrieved_chunks>\n{context}\n</retrieved_chunks>\n"
            "Response format:\n"
            "Answer: <your answer>\n"
            "Sources:\n"
            "- path [start:end]\n"
        )

    def augument(self, query: str, k: int = 5, path: Path = INDEX_DIR) -> str:
        question = _as_question(query)
        chunks = self._retrieve(question, k=k, path=path)
        context = "\n\n".join(
            f"{src.file_path} "
            f"[{src.first_character_index}:{src.last_character_index}]\n"
            f"{src.text}"
            for src in chunks
        )
        return self._prompt(question.question, context)

    def _generate(self, prompt: str) -> str:
        response = generate(
            model=MODEL,
            prompt=prompt,
            think=False,
            options={"num_predict": 256},
        )
        return response.response

    def answer(self, query: str, k: int = 5):
        print(self._generate(self.augument(query, k)))

    def answer_dataset(
        self,
        student_search_results_path: Path = SEARCH_OUT_DIR / "dataset_docs_public.json",
        save_directory: Path = SAVE_ANS,
    ):
        path = Path(student_search_results_path)
        results = StudentSearchResults.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        answers: list[MinimalAnswer] = []
        for item in tqdm(results.search_results, desc="Answering"):
            prompt = self._prompt(
                item.question, self._sources_text(item.retrieved_sources)
            )
            answers.append(MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                answer=self._generate(prompt),
                retrieved_sources=item.retrieved_sources,
            ))

        output = StudentSearchResultsAndAnswer(search_results=answers, k=results.k)
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        out_file = save_directory / path.name
        out_file.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {out_file}")


def main() -> None:
    rag = Rag()
    fire.Fire(rag)


if __name__ == "__main__":

    main()