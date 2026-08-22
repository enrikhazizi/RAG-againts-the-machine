install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb src

lint:
	flake8 src
	mypy

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

temp:
	cp -r ../RAG-againts-the-machine /goinfre/ehazizi
