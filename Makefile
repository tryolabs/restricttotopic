dev:
	pip install -e ".[dev]"

lint:
	ruff check .

type:
	pyright validator

qa:
	make lint
	make type