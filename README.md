*This activity has been created as part of the 42 curriculum by ehazizi.*

## Description

**RAG against the Machine** is a retrieval-augmented generation pipeline over the
vLLM **0.10.1** source tree. It chunks Python and Markdown (plus a few text
files), builds a **BM25** index, retrieves ranked source locations for
benchmark questions, optionally generates grounded answers with a small local
LLM, and reports **recall@k** against labeled gold spans.

The project satisfies the 42 subject requirements:

- Two chunking strategies (AST for `.py`, headings for `.md` / `.txt`)
- Maximum chunk size **2000** characters
- CLI commands: `index`, `search`, `search_dataset`, `answer`,
  `answer_dataset`, `evaluate`
- Pydantic data models compatible with the moulinette JSON schema
- Local recall evaluation with **IoU ≥ 0.05** on matching `file_path`

Official grading uses the moulinette under `tests/moulinette_pkg/`. Student code
must **not** import or call the moulinette.

## Instructions

### Setup

Requirements: Python **3.10+**, [uv](https://docs.astral.sh/uv/), and Ollama
(only for answer commands).

```bash
make install
ollama pull Qwen3:0.6b
```

### 1. Index the corpus

Run once before any search or answer command. This walks
`data/raw/vllm-0.10.1`, chunks files, tokenizes them, and saves a BM25 index
under `data/processed/vllm/` (gitignored).

```bash
uv run python -m src index --max_chunk_size 2000
```

Optional flags: `--corpus_path`, `--path` (index output directory).

### 2. Search

**Single question** — prints ranked `file_path [start:end]` lines:

```bash
uv run python -m src search "How do I load a LoRA adapter?" --k 5
```

**Full dataset** — writes a `StudentSearchResults` JSON file:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Repeat for `dataset_code_public.json` with a separate `--save_directory`.
Always use **`--k 10`** so recall@5 and recall@10 can be measured.

Datasets live under `data/datasets/UnansweredQuestions/` and
`data/datasets/AnsweredQuestions/`.

### 3. Answer (optional)

Requires a running Ollama server and the `Qwen3:0.6b` model.

```bash
uv run python -m src answer "How do I load a LoRA adapter?" --k 5

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

`answer_dataset` reads search JSON, rebuilds chunk text from disk, calls the
model, and writes `StudentSearchResultsAndAnswer` JSON.

### 4. Evaluate retrieval

Compare search output against **AnsweredQuestions** gold labels (not
UnansweredQuestions):

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

Official moulinette check:

```bash
./tests/moulinette_pkg/moulinette-fedora evaluate_student_search_results \
  data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000 --threshold 0.80
```

Pass criteria on the public sets:

| Dataset | Metric    | Threshold |
|---------|-----------|-----------|
| Docs    | Recall@5  | ≥ 80%     |
| Code    | Recall@5  | ≥ 50%     |

### 5. Lint

```bash
make lint
```

Other Makefile targets: `run`, `debug` (`pdb -m src`), `clean`.

### Example usage (full workflow)

```bash
# setup
make install
ollama pull Qwen3:0.6b

# index
uv run python -m src index --max_chunk_size 2000

# search public docs + code (for moulinette scoring)
uv run python -m src search_dataset \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/AnsweredQuestions

uv run python -m src search_dataset \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/AnsweredQuestions

# local + official eval (docs)
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10

./tests/moulinette_pkg/moulinette-fedora evaluate_student_search_results \
  data/output/search_results/AnsweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000 --threshold 0.80

# optional: generate answers for unanswered set
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## Architecture

```text
data/raw/vllm-0.10.1/
        │
        ▼
  DatasetChunker (src/chunking.py)
        │
        ▼
  BM25 index (data/processed/vllm/)
        │
        ├── search / search_dataset ──► StudentSearchResults JSON
        │
        ├── answer / answer_dataset ──► StudentSearchResultsAndAnswer JSON
        │
        └── evaluate ──► Recall@1/3/5/10 (stdout)
```

- **CLI**: `src/__main__.py` — Fire wrapper around the `Rag` class
- **Models**: `src/data_models/models.py` — Pydantic schemas for all JSON I/O
- **Chunking**: `src/chunking.py` — Python AST + Markdown heading splitters
- **Generation**: Ollama + `Qwen3:0.6b`, thinking disabled, 256 token cap

## Chunking strategy

| Extension   | Method |
|-------------|--------|
| `.py`       | Parse with `ast`; emit chunks for functions, classes, and module-level statements; hard-cut oversized units at `max_chunk_size` |
| `.md`       | Split on `#` headings; hard-cut long sections |
| `.txt`      | Same as Markdown (covers `CMakeLists.txt` in the docs gold set) |

Each chunk records `source`, `start`, `end`, and `text`. Stored paths are
relative to the project root (`data/raw/vllm-0.10.1/...`) so they match
moulinette gold labels.

## Retrieval strategy

- **Engine**: `bm25s` BM25 with English Snowball stemming
- **Tokenization**: custom `code_split` — lowercase tokens plus snake_case
  pieces (`load_lora` → `load`, `lora`)
- **Indexing boost**: file stem tokens are prepended to each chunk at index time
- **Dataset filter**: filenames containing `dataset_docs` restrict hits to
  `.md` and `CMakeLists.txt`; `dataset_code` restricts to `.py`. The retriever
  oversamples 50 BM25 hits, filters by extension, then keeps the top `k`. This
  avoids code chunks ranking above documentation answers (and vice versa).

## Performance

On a typical laptop CPU:

- Indexing ~**41k** chunks: under one minute
- `search_dataset` over 100 questions: a few seconds (index loaded once)
- `answer` per question: depends on Ollama; capped at 256 generated tokens

Measured public scores after indexing:

- Docs Recall@5: **84%** (moulinette PASS ≥ 80%)
- Code Recall@5: **65%** (moulinette PASS ≥ 50%)

## Design decisions

- **Pydantic models** kept minimal for moulinette compatibility; extra fields
  are allowed but required fields are unchanged.
- **No moulinette import** — `evaluate` reimplements IoU + recall locally for
  iteration during development.
- **`_display_path`** normalizes chunk paths to project-relative strings.
- **Edge cases** (empty query, `k=0`, bad `k`, missing/malformed files) print a
  short message and return without traceback.

## Challenges

- Gold spans are character ranges inside large files; retrieving the correct
  *section* is harder than retrieving the correct file. Docs/code filtering and
  BM25 oversampling were required to reach the public recall bars.
- IoU at 5% means chunk boundaries must overlap gold labels, not just share a
  file path.
- Three documentation answers point at `CMakeLists.txt`, which required adding
  `.txt` chunking.

## Resources

- vLLM 0.10.1 corpus: `data/raw/vllm-0.10.1/`
- Public question sets: `data/datasets/`
- Moulinette docs: `tests/moulinette_pkg/README.md`
- [bm25s](https://github.com/xhluca/bm25s) — BM25 implementation
- [Ollama](https://ollama.com/) — local LLM server for Qwen3-0.6B
- [Google Fire](https://github.com/google/python-fire) — CLI framework
- [Pydantic](https://docs.pydantic.dev/) — data validation

### AI usage

- Debugging mypy/flake8 compliance
- Analyzing recall failures

All code was reviewed, linted, and validated against the public moulinette
before submission.
