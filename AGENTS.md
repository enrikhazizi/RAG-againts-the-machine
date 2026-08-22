# AGENTS.md

## Cursor Cloud specific instructions

This is a Python RAG (Retrieval-Augmented Generation) project managed with [`uv`](https://docs.astral.sh/uv/). It targets Python 3.14 (see `.python-version`) and `uv` provisions that interpreter automatically — do not rely on the system `python3` (3.12).

Dependencies (installed by the startup update script via `uv sync`) include the heavy `vllm` + `torch` stack, but the current `src/` code does not import them yet.

### Services / components

- `src/` — the student RAG library/CLI (entrypoint prints a placeholder). Core code lives in `src/data_models` (pydantic models for the dataset) and `src/retriver_metric` (recall/precision `Metric_tester`).
- `tests/moulinette_pkg/moulinette-ubuntu` (and `-fedora`) — prebuilt PyInstaller binaries that are the reference **evaluation tool**. See `tests/moulinette_pkg/README.md` for its full CLI. It evaluates student search results against a ground-truth dataset (`data/dataset/...`) and reports Recall@k.

### Run / lint / test — non-obvious notes

- Run the app with `uv run python -m src.main` (or `uv run python src/main.py`). `make run` (`uv run python -m src`) is BROKEN because `src/` has no `__main__.py`.
- Lint: `make lint` calls `flake8 src` and `mypy`, but neither is a declared dependency. Run them ad hoc without polluting the project via `uvx flake8 src` / `uvx mypy src`.
- Tests: there are no `pytest` tests; `pytest` is not installed and `tests/` only holds the moulinette binaries. Functional "testing" means running the moulinette evaluation.
- The package `__init__.py` files use flat imports (e.g. `from models import ...`) rather than relative imports, so importing `src.data_models` directly fails; add `src/data_models` and `src/retriver_metric` to `sys.path` when exercising them from a standalone script.

### Quick end-to-end evaluation smoke test

```bash
# moulinette expects a student search-results JSON; a perfect one can be derived
# from the ground truth. Then:
./tests/moulinette_pkg/moulinette-ubuntu evaluate_student_search_results \
    <student_results.json> \
    data/dataset/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --threshold 0.80
```
