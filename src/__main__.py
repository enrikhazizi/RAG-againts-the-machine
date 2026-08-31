# from vllm import LLM, SamplingParams
from chunking import DatasetChunker
from bm25s.tokenization import Tokenizer
from pathlib import Path
from tqdm import tqdm
import json
import fire
import bm25s

INDEX_DIR = Path("data/processed/vllm")
QUERY_DIR = Path("data/dataset/UnansweredQuestions")

def _tokenizer() -> Tokenizer:
    return Tokenizer(stemmer=None, stopwords=[], splitter=lambda x: x.split())


def _display_path(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return path

def _load_json(path: Path) -> 

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
            tokens = tokenizer.tokenize(
                [chunk["text"]], update_vocab=True, 
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

    def search(self, query: str, k: int = 5):
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
                f"{_display_path(doc['source'])}"
                f"[{doc['start']}:{doc['end']}]\n"
                )
  
        return "".join(result)

    def augument(self, query: str) -> str:
        return (
            f"User quesition : {query}"
            "Context files and code to use for awnsering the user question"
            f"{self.search(query, 5)}"
        )
    
    def bulk_gen(self, path: Path)

if __name__ == "__main__":
    fire.Fire(Rag)
