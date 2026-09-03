# from vllm import LLM, SamplingParams
from data_models import FullSource
from chunking import DatasetChunker
from bm25s.tokenization import Tokenizer
from pathlib import Path
from tqdm import tqdm
import Stemmer
import json
import re
import fire
import bm25s

INDEX_DIR = Path("data/processed/vllm")
QUERY_DIR = Path("data/dataset/UnansweredQuestions")


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


def _load_json(path: Path):
    
    dataset_code = []
    dataset_docs = []
    for file in path.glob("*"):
        if file.is_file():
            with open(file, "r", encoding="utf-8") as f:
                print(json.load(f))


class Rag():

    # llm = LLM(model="Qwen/Qwen3-0.6B")
    # sampling_params = SamplingParams(
    #     temperature=0.2,
    #     max_tokens=256
    #     )

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

    def search(self, query: str, k: int = 5, inside_use: bool = False) -> str:
        result = []
        try:
            retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
        except FileNotFoundError as e:
            print(e)
            return

        query_tokens = _tokenizer().tokenize(
            [query],
            update_vocab=True,
            return_as="string",
            show_progress=False,
        )
        docs, scores = retriever.retrieve(query_tokens, k=k)

        for rank, (doc, score) in enumerate(zip(docs[0], scores[0]), start=1):
            # print(f"--- {rank}  score={float(score):.4f} ---")

            result.append(
                FullSource(
                    file_path=_display_path(doc['source']),
                    first_character_index=doc['start'],
                    last_character_index=doc['end'],
                    text=doc['text']
                )
            )

        return "\n".join(
            f"{src.text}" if inside_use else
            f"{src.file_path}:[{src.first_character_index}]"
            f"-[{src.last_character_index}]"
            for src in result
        )

    def augument(self, query: str) -> str:
        context = self.search(query, k=5)
        if not context:
            context = "(no chunks retrieved — index may be missing)"
        return (
            f"""You are answering questions about the vLLM codebase "
            "from retrieved source chunks.
            Rules:
            - Use only the chunks below. Do not "
            "invent files, functions, or behavior.
            - If the chunks do not contain enough evidence, say s"
            "o and answer only what they support.
            - Cite every claim with file path and character range [start:end].
            <question>
            {query}
            </question>
            <retrieved_chunks>
            {context}
            </retrieved_chunks>
            Response format:
            Answer: <your answer>
            Sources:
            - path [start:end]
            """
            )


def main() -> None:
    rag = Rag()
    fire.Fire(rag)


if __name__ == "__main__":

    _load_json(QUERY_DIR)
