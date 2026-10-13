.PHONY: demo test test-live check fmt install

install:
	uv sync

demo:
	uv run drift demo

test:
	uv run pytest

test-live:
	DRIFT_LIVE_OLLAMA=1 uv run pytest -m live -p no:cacheprovider

check:
	uv run ruff check coursedrift tests
	uv run ruff format --check coursedrift tests
	uv run mypy coursedrift

fmt:
	uv run ruff format coursedrift tests
	uv run ruff check --fix coursedrift tests
