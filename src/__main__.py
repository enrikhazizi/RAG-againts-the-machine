from vllm import LLM, SamplingParams
from data_models import (FullSource, UnansweredQuestion,
                         RagDataset, StudentSearchResults,
                         MinimalSource, MinimalSearchResults,
                         StudentSearchResultsAndAnswer)
from chunking import DatasetChunker
from bm25s.tokenization import Tokenizer
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
    with open(path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    questions = [
        UnansweredQuestion(
            question_id=data["question_id"],
            question=data["question"],
        )
        for data in json_file["rag_questions"]
    ]
    return RagDataset(rag_questions=questions)


class Rag():

    llm = LLM(model="Qwen/Qwen3-0.6B", avx512f)
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=256
        )

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
        save_directory: Path = SEARCH_OUT_DIR,
        write: bool = True
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
        if write:
            return results

    def augument(self, query: str, k: int = 5, path: Path = INDEX_DIR) -> str:
        question = _as_question(query)
        chunks = self._retrieve(question, k=k, path=path)
        if not chunks:
            context = "(no chunks retrieved — index may be missing)"
        else:
            context = "\n\n".join(
                f"{src.file_path} "
                f"[{src.first_character_index}:{src.last_character_index}]\n"
                f"{src.text}"
                for src in chunks
            )
        return (
            "You are answering questions about the vLLM codebase "
            "from retrieved source chunks.\n"
            "Rules:\n"
            "- Use only the chunks below. Do not "
            "invent files, functions, or behavior.\n"
            "- If the chunks do not contain enough evidence, say so "
            "and answer only what they support.\n"
            "- Cite every claim with file path and character range [start:end].\n"
            f"<question>\n{question.question}\n</question>\n"
            f"<retrieved_chunks>\n{context}\n</retrieved_chunks>\n"
            "Response format:\n"
            "Answer: <your answer>\n"
            "Sources:\n"
            "- path [start:end]\n"
        )

    def answer(self, query: str, k: int):
        question = self.augument(query, k)
        ans = self.llm.generate(prompts=question, sampling_params=self.sampling_params)
        print(ans)

    def answer_dataset(self, question_path: Path = QUERY_DIR, k: int = 5, path: Path = INDEX_DIR) -> StudentSearchResultsAndAnswer:
        ans = []
        dataset = _load_json(question_path)
        for question in dataset.rag_questions:
            print(question)


def main() -> None:
    rag = Rag()
    fire.Fire(rag)


if __name__ == "__main__":

    main()