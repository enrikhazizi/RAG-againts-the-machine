"""Fire CLI for indexing, retrieval, answering, and recall@k."""
from .data_models import (
    AnsweredQuestion,
    FullSource,
    UnansweredQuestion,
    RagDataset,
    StudentSearchResults,
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from .chunking import DatasetChunker
from bm25s.tokenization import Tokenizer
from ollama import generate
from pathlib import Path
from pydantic import ValidationError
from tqdm import tqdm
from typing import Any
import Stemmer
import json
import re
import fire
import bm25s

INDEX_DIR = Path("data/processed/vllm")
QUERY_DIR = Path("data/datasets/UnansweredQuestions")
SEARCH_OUT_DIR = Path(
    "data/output/search_results/UnansweredQuestions"
)
SAVE_ANS = Path(
    "data/output/search_results_and_answer/UnansweredQuestions"
)
MODEL = "Qwen3:0.6b"
IOU_THRESHOLD = 0.05
RECALL_KS = (1, 3, 5, 10)
RETRIEVE_OVERSAMPLE = 50


def _tokenizer() -> Tokenizer:
    """Build the BM25 tokenizer used at index and query time.

    Returns:
        A tokenizer with an English stemmer and ``code_split``.
    """
    return Tokenizer(
        stemmer=Stemmer.Stemmer("english"),
        stopwords=[],
        splitter=code_split,
    )


def code_split(text: str) -> list[str]:
    """Split text into lowercase word tokens, cutting identifiers apart.

    Args:
        text: Raw chunk or query text.

    Returns:
        Lowercased tokens, including snake_case pieces.
    """
    tokens: list[str] = []
    for word in re.split(r"[^0-9a-zA-Z_]+", text):
        if not word:
            continue
        word_lower = word.lower()
        tokens.append(word_lower)
        if "_" in word_lower:
            snake_pieces = [p for p in word_lower.split("_") if p]
            tokens.extend(snake_pieces)
    return tokens


def _infer_source_kind(dataset_path: str | Path) -> str | None:
    """Guess docs vs code retrieval scope from a dataset file name.

    Args:
        dataset_path: Path to a dataset JSON file.

    Returns:
        ``"docs"``, ``"code"``, or None when unknown.
    """
    name = Path(dataset_path).name.lower()
    if "dataset_docs" in name or "_docs_" in name:
        return "docs"
    if "dataset_code" in name or "_code_" in name:
        return "code"
    return None


def _source_allowed(file_path: str, kind: str | None) -> bool:
    """Return True when a chunk path matches the dataset kind filter.

    Args:
        file_path: Chunk path stored in the index.
        kind: ``"docs"``, ``"code"``, or None for no filtering.

    Returns:
        Whether the path should be kept for this dataset kind.
    """
    if kind is None:
        return True
    path = Path(file_path)
    if kind == "docs":
        return path.suffix == ".md" or path.name == "CMakeLists.txt"
    if kind == "code":
        return path.suffix == ".py"
    return True


def _display_path(path: str) -> str:
    """Make a corpus path relative to the project root when possible.

    Args:
        path: Absolute or relative file path stored on a chunk.

    Returns:
        A path the grader can compare verbatim to the corpus.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return path


def _as_int(value: object, name: str) -> int | None:
    """Parse a CLI integer, or print an error and return None.

    Args:
        value: Raw Fire argument.
        name: Name used in the error message.

    Returns:
        The integer, or None if ``value`` is not an int-like value.
    """
    if isinstance(value, bool):
        print(f"Invalid {name}={value!r}: expected an integer.")
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("+-").isdigit():
            return int(stripped)
    print(f"Invalid {name}={value!r}: expected an integer.")
    return None


def _as_question(query: UnansweredQuestion | str) -> UnansweredQuestion:
    """Normalize a CLI query into an UnansweredQuestion.

    Args:
        query: A question object or a raw string.

    Returns:
        An UnansweredQuestion with a ``question`` field set.
    """
    if isinstance(query, UnansweredQuestion):
        return query
    return UnansweredQuestion(question=str(query))


def _is_blank(text: str) -> bool:
    """Return True when ``text`` is empty or only whitespace.

    Args:
        text: Query or question string.

    Returns:
        True if there is nothing to search.
    """
    return not str(text).strip()


def _iou(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> float:
    """Intersection-over-union of two character ranges.

    Args:
        a_start: Start of the first span.
        a_end: End of the first span.
        b_start: Start of the second span.
        b_end: End of the second span.

    Returns:
        IoU in ``[0, 1]``. Zero when the union is empty.
    """
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _source_found(
    truth: MinimalSource, retrieved: list[MinimalSource]
) -> bool:
    """Return True if any retrieved span overlaps ``truth`` enough.

    Args:
        truth: Gold source.
        retrieved: Student sources to scan.

    Returns:
        True when the same file has IoU >= ``IOU_THRESHOLD``.
    """
    for src in retrieved:
        if src.file_path != truth.file_path:
            continue
        score = _iou(
            src.first_character_index,
            src.last_character_index,
            truth.first_character_index,
            truth.last_character_index,
        )
        if score >= IOU_THRESHOLD:
            return True
    return False


def _recall_at_k(
    truths: list[MinimalSource],
    retrieved: list[MinimalSource],
    k: int,
) -> float:
    """Share of gold sources found in the top-k retrieved results.

    Args:
        truths: Gold sources for one question.
        retrieved: Ranked student sources.
        k: Cut-off on the student list.

    Returns:
        Recall in ``[0, 1]``, or 0.0 when there is no gold.
    """
    if not truths or k <= 0:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for item in truths if _source_found(item, top))
    return found / len(truths)


def _load_json(path: Path) -> RagDataset | None:
    """Load one dataset file or every JSON file in a directory.

    Args:
        path: Dataset file or directory.

    Returns:
        A RagDataset, or None when the path is missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return None

    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        print(f"No JSON files in {path}")
        return None

    questions: list[AnsweredQuestion | UnansweredQuestion] = []
    for file in files:
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
            dataset = RagDataset.model_validate(raw)
        except json.JSONDecodeError as exc:
            print(f"Malformed JSON in {file}: {exc}")
            return None
        except (OSError, ValidationError, TypeError, ValueError) as exc:
            print(f"Invalid dataset in {file}: {exc}")
            return None
        questions.extend(dataset.rag_questions)
    return RagDataset(rag_questions=questions)


def _load_search_results(
    path: Path,
) -> StudentSearchResults | None:
    """Load a StudentSearchResults JSON file.

    Args:
        path: Path written by ``search_dataset``.

    Returns:
        The parsed model, or None on missing / malformed input.
    """
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        parsed = StudentSearchResults.model_validate(raw)
    except json.JSONDecodeError as exc:
        print(f"Malformed JSON in {path}: {exc}")
        return None
    except (OSError, ValidationError, TypeError, ValueError) as exc:
        print(f"Invalid search results in {path}: {exc}")
        return None
    if not isinstance(parsed, StudentSearchResults):
        print(f"Invalid search results in {path}")
        return None
    return parsed


class Rag:
    """Retrieval-Augmented Generation over the vLLM 0.10.1 source tree.

    Pipeline: `index` chunks the corpus and builds a BM25 index, `search` /
    `search_dataset` retrieve source locations, `answer` / `answer_dataset`
    feed those chunks to Qwen3-0.6B to produce grounded answers.

    Run `uv run python -m src <command> --help` for per-command options.
    """

    def index(
        self,
        max_chunk_size: int = 2000,
        corpus_path: str | Path = "data/raw/vllm-0.10.1",
        path: str | Path = INDEX_DIR,
    ) -> None:
        """Chunk the corpus and build the BM25 index.

        Walks the vLLM tree, splits every .py file by AST node
        (functions, classes, module-level statements) and every .md file
        by heading, then tokenizes each chunk and persists a BM25 index.
        Run this once before any search or answer command.

        Args:
            max_chunk_size: Upper bound on chunk length in characters.
                Larger logical units are hard-cut into windows of this
                size. Must be between 1 and 2000.
            corpus_path: Root of the files to ingest.
            path: Directory where the BM25 index is saved.
        """
        size = _as_int(max_chunk_size, "max_chunk_size")
        if size is None:
            return
        if size < 1 or size > 2000:
            print(
                "max_chunk_size must be between 1 and 2000 "
                f"(got {size})."
            )
            return

        chunker = DatasetChunker(max_chunk=size)
        my_dir = Path(corpus_path)
        if not my_dir.exists():
            print(f"Corpus not found: {my_dir}")
            return

        corpus = chunker.chunk_directory(my_dir)
        if not corpus:
            print(
                "No chunks produced. Check the corpus path "
                "and chunk size."
            )
            return

        tokenizer = _tokenizer()
        all_ids: list[Any] = []
        for chunk in tqdm(corpus, desc="Tokenizing chunks", unit=" chunk"):
            path_words = Path(chunk["source"]).stem.replace("-", " ")
            tokens = tokenizer.tokenize(
                [path_words + " " + chunk["text"]],
                update_vocab=True,
                return_as="ids",
                show_progress=False,
            )
            all_ids.extend(tokens)

        corpus_tokens = tokenizer.to_tokenized_tuple(all_ids)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        retriever.save(save_path)
        print(
            f"Ingestion complete! Indexed {len(corpus)} chunks. "
            f"Indices saved under {save_path}"
        )

    def _load_retriever(self, path: Path) -> Any:
        """Load a saved BM25 index, or None if it is missing.

        Args:
            path: Directory holding the saved BM25 index.

        Returns:
            The loaded retriever, or None on failure.
        """
        path = Path(path)
        if not path.exists():
            print(
                f"Index not found: {path}. Run `index` first."
            )
            return None
        try:
            return bm25s.BM25.load(path, load_corpus=True)
        except Exception as exc:
            print(f"Failed to load index at {path}: {exc}")
            return None

    def _retrieve(
        self,
        query: UnansweredQuestion,
        k: int = 5,
        path: Path = INDEX_DIR,
        retriever: Any = None,
        source_kind: str | None = None,
    ) -> list[FullSource]:
        """Run one BM25 query and return the top-k chunks with their text.

        Args:
            query: The question to search for.
            k: Number of chunks to return. ``k <= 0`` yields no chunks.
            path: Directory holding the saved BM25 index.
            retriever: Already-loaded index to reuse across many queries.
                When None the index is loaded from `path`.
            source_kind: Optional ``"docs"`` or ``"code"`` filter inferred
                from the dataset file name.

        Returns:
            Ranked chunks as FullSource (path, character range and text).
        """
        if k <= 0 or _is_blank(query.question):
            return []

        if retriever is None:
            retriever = self._load_retriever(Path(path))
            if retriever is None:
                return []

        fetch_k = k
        if source_kind is not None:
            fetch_k = max(k, RETRIEVE_OVERSAMPLE)

        try:
            query_tokens = _tokenizer().tokenize(
                [query.question],
                update_vocab=True,
                return_as="string",
                show_progress=False,
            )
            docs, _scores = retriever.retrieve(query_tokens, k=fetch_k)
        except Exception as exc:
            print(f"Retrieval failed: {exc}")
            return []

        if docs is None or len(docs) == 0:
            return []

        retrieved: list[FullSource] = []
        for doc in docs[0]:
            if not isinstance(doc, dict):
                continue
            file_path = _display_path(str(doc.get("source", "")))
            if not _source_allowed(file_path, source_kind):
                continue
            retrieved.append(FullSource(
                file_path=file_path,
                first_character_index=int(doc.get("start", 0)),
                last_character_index=int(doc.get("end", 0)),
                text=str(doc.get("text", "")),
            ))
            if len(retrieved) >= k:
                break
        return retrieved

    def search(
        self,
        query: str,
        k: int = 5,
        path: str | Path = INDEX_DIR,
    ) -> None:
        """Print the top-k source locations for a single question.

        Each line is `file_path [first_character_index:last_character_index]`,
        ranked by BM25 score. Paths are relative to the project root and
        match the corpus exactly.

        Args:
            query: The natural-language question to search for.
            k: Number of results to print.
            path: Directory holding the BM25 index built by `index`.

        Example:
            uv run python -m src search \\
                "How do I load a LoRA adapter?" --k 5
        """
        k_value = _as_int(k, "k")
        if k_value is None:
            return
        if k_value < 0:
            print(f"Invalid k={k_value}: must be >= 0.")
            return
        question = _as_question(query)
        if _is_blank(question.question):
            print("Empty query: nothing to search.")
            return
        if k_value == 0:
            print("k=0: no sources requested.")
            return
        for source in self._retrieve(
            question, k=k_value, path=Path(path)
        ):
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str | Path,
        k: int = 5,
        path: str | Path = INDEX_DIR,
        save_directory: str | Path = SEARCH_OUT_DIR,
    ) -> None:
        """Search every question in a dataset and write JSON.

        Loads the index once, retrieves the top-k sources for each
        question in the dataset, and writes the results next to the
        original file name inside `save_directory`. This output is
        what the moulinette scores.

        Args:
            dataset_path: JSON file with a `rag_questions` list
                (UnansweredQuestions or AnsweredQuestions format).
            k: Number of sources to keep per question.
            path: Directory holding the BM25 index built by `index`.
            save_directory: Output folder. Scope it by dataset
                because the public docs and code datasets share
                file names.

        Example:
            uv run python -m src search_dataset \\
                --dataset_path <dataset.json> \\
                --k 10 \\
                --save_directory \\
                data/output/search_results/UnansweredQuestions
        """
        k_value = _as_int(k, "k")
        if k_value is None:
            return
        if k_value < 0:
            print(f"Invalid k={k_value}: must be >= 0.")
            return

        dataset = _load_json(Path(dataset_path))
        if dataset is None:
            return

        source_kind = _infer_source_kind(dataset_path)

        retriever = None
        if k_value > 0:
            retriever = self._load_retriever(Path(path))
            if retriever is None:
                return

        results: list[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            retrieved = self._retrieve(
                question,
                k=k_value,
                retriever=retriever,
                source_kind=source_kind,
            )
            results.append(MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=src.file_path,
                        first_character_index=(
                            src.first_character_index
                        ),
                        last_character_index=(
                            src.last_character_index
                        ),
                    )
                    for src in retrieved
                ],
            ))

        output = StudentSearchResults(
            search_results=results, k=k_value
        )
        save_dir = Path(save_directory)
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_file = save_dir / Path(dataset_path).name
            out_file.write_text(
                output.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:
            print(f"Could not write results: {exc}")
            return
        print(f"Saved student_search_results to {out_file}")

    def _sources_text(self, sources: list[MinimalSource]) -> str:
        """Rebuild chunk text from disk for sources that carry no text.

        Args:
            sources: Locations (path + character range) to read back.

        Returns:
            The chunks joined by blank lines, each prefixed with its
            location. Missing files are skipped with a note.
        """
        parts: list[str] = []
        for src in sources:
            try:
                text = Path(src.file_path).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError as exc:
                parts.append(
                    f"{src.file_path} (unreadable: {exc})"
                )
                continue
            chunk = text[
                src.first_character_index:src.last_character_index
            ]
            parts.append(
                f"{src.file_path} "
                f"[{src.first_character_index}:"
                f"{src.last_character_index}]\n"
                f"{chunk}"
            )
        return "\n\n".join(parts)

    def _prompt(self, question: str, context: str) -> str:
        """Build the grounded-answer prompt sent to the model.

        Args:
            question: The user question.
            context: Retrieved chunks, each prefixed with its location.

        Returns:
            The full prompt string.
        """
        if not context:
            context = (
                "(no chunks retrieved — index may be missing)"
            )
        return (
            "You are answering questions about the vLLM codebase "
            "from retrieved source chunks.\n"
            "Rules:\n"
            "- Use only the chunks below. Do not "
            "invent files, functions, or behavior.\n"
            "- If the chunks do not contain enough evidence, "
            "say so and answer only what they support.\n"
            "- Cite every claim with file path and character"
            " range [start:end].\n"
            f"<question>\n{question}\n</question>\n"
            f"<retrieved_chunks>\n{context}\n</retrieved_chunks>\n"
            "Response format:\n"
            "Answer: <your answer>\n"
            "Sources:\n"
            "- path [start:end]\n"
        )

    def augument(
        self, query: str, k: int = 5, path: str | Path = INDEX_DIR
    ) -> str:
        """Retrieve the top-k chunks and return the prompt built from them.

        Useful for inspecting exactly what the model will see before
        calling `answer`. Nothing is sent to the model.

        Args:
            query: The natural-language question.
            k: Number of chunks to place in the prompt.
            path: Directory holding the BM25 index built by `index`.

        Returns:
            The complete prompt string, including the retrieved chunks.
        """
        k_value = _as_int(k, "k")
        if k_value is None:
            k_value = 0
        question = _as_question(query)
        if _is_blank(question.question):
            return self._prompt(question.question, "")
        chunks = self._retrieve(
            question, k=k_value, path=Path(path)
        )
        context = "\n\n".join(
            f"{src.file_path} "
            f"[{src.first_character_index}:"
            f"{src.last_character_index}]\n"
            f"{src.text}"
            for src in chunks
        )
        return self._prompt(question.question, context)

    def _generate(self, prompt: str) -> str:
        """Send a prompt to Qwen3-0.6B via Ollama and return the text.

        Thinking is disabled and output is capped at 256 tokens to keep
        CPU latency low.

        Args:
            prompt: The full prompt produced by `_prompt`.

        Returns:
            The model's response text, or an error note on failure.
        """
        try:
            response = generate(
                model=MODEL,
                prompt=prompt,
                think=False,
                options={"num_predict": 256},
            )
        except Exception as exc:
            return f"(generation failed: {exc})"
        text = getattr(response, "response", "")
        if isinstance(text, str):
            return text
        return "" if text is None else str(text)

    def answer(
        self,
        query: str,
        k: int = 5,
        path: str | Path = INDEX_DIR,
    ) -> None:
        """Answer a single question using the top-k retrieved chunks.

        Retrieves k chunks, builds a grounded prompt, and prints the
        model's answer. Requires the index (see `index`) and a running
        Ollama server with the Qwen3-0.6B model pulled.

        Args:
            query: The natural-language question to answer.
            k: Number of chunks to pass to the model as context.
            path: Directory holding the BM25 index built by `index`.

        Example:
            uv run python -m src answer \\
                "How do I load a LoRA adapter?" --k 5
        """
        k_value = _as_int(k, "k")
        if k_value is None:
            return
        if k_value < 0:
            print(f"Invalid k={k_value}: must be >= 0.")
            return
        question = _as_question(query)
        if _is_blank(question.question):
            print("Empty query: nothing to answer.")
            return
        print(self._generate(self.augument(query, k_value, path)))

    def answer_dataset(
        self,
        student_search_results_path: str | Path = (
            SEARCH_OUT_DIR / "dataset_docs_public.json"
        ),
        save_directory: str | Path = SAVE_ANS,
    ) -> None:
        """Generate an answer for every question in a search JSON.

        Reads a StudentSearchResults JSON produced by `search_dataset`,
        rebuilds each question's context from the stored source
        locations (no new retrieval), asks the model, and writes a
        StudentSearchResultsAndAnswer JSON with the same file name
        into `save_directory`.

        Args:
            student_search_results_path: StudentSearchResults JSON
                written by `search_dataset`.
            save_directory: Output folder. Scope it by dataset.

        Example:
            uv run python -m src answer_dataset \\
                --student_search_results_path <results.json> \\
                --save_directory \\
                data/output/search_results_and_answer/UnansweredQuestions
        """
        path = Path(student_search_results_path)
        results = _load_search_results(path)
        if results is None:
            return
        answers: list[MinimalAnswer] = []
        for item in tqdm(results.search_results, desc="Answering"):
            prompt = self._prompt(
                item.question,
                self._sources_text(item.retrieved_sources),
            )
            answers.append(MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                answer=self._generate(prompt),
                retrieved_sources=item.retrieved_sources,
            ))

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=results.k
        )
        save_dir = Path(save_directory)
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_file = save_dir / path.name
            out_file.write_text(
                output.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:
            print(f"Could not write answers: {exc}")
            return
        print(
            "Saved student_search_results_and_answer "
            f"to {out_file}"
        )

    def evaluate(
        self,
        student_search_results_path: str | Path,
        dataset_path: str | Path,
        k: int = 10,
    ) -> None:
        """Report recall@k against a ground-truth dataset.

        A gold source counts as found when a retrieved result is in the
        same file and overlaps its character range with IoU >= 0.05.
        This is for local iteration; the defense uses the moulinette.

        Args:
            student_search_results_path: StudentSearchResults JSON.
            dataset_path: AnsweredQuestions ground-truth JSON.
            k: Maximum number of student sources to consider.
        """
        k_value = _as_int(k, "k")
        if k_value is None:
            return
        if k_value < 0:
            print(f"Invalid k={k_value}: must be >= 0.")
            return

        student = _load_search_results(
            Path(student_search_results_path)
        )
        gold = _load_json(Path(dataset_path))
        if student is None or gold is None:
            return

        gold_by_id: dict[str, AnsweredQuestion] = {}
        for item in gold.rag_questions:
            if isinstance(item, AnsweredQuestion):
                gold_by_id[item.question_id] = item
        if not gold_by_id:
            print("Ground-truth dataset has no answered questions.")
            return

        student_by_id = {
            item.question_id: item
            for item in student.search_results
        }

        cutoffs = [n for n in RECALL_KS if n <= k_value]
        if k_value not in cutoffs:
            cutoffs.append(k_value)
        scores: dict[int, list[float]] = {n: [] for n in cutoffs}

        for question_id, truth in gold_by_id.items():
            retrieved: list[MinimalSource] = []
            hit = student_by_id.get(question_id)
            if hit is not None:
                retrieved = hit.retrieved_sources
            for cutoff in cutoffs:
                scores[cutoff].append(
                    _recall_at_k(truth.sources, retrieved, cutoff)
                )

        n_q = len(gold_by_id)
        print(f"Evaluated {n_q} questions.")
        parts = []
        for cutoff in cutoffs:
            values = scores[cutoff]
            mean = sum(values) / len(values) if values else 0.0
            parts.append(f"Recall@{cutoff}: {mean:.3f}")
        print(" ".join(parts))


def main() -> None:
    """Entry point: expose the Rag class as a Fire CLI."""
    try:
        fire.Fire(Rag(), name="src")
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
