.PHONY: demo test test-live eval check fmt install

install:
	uv sync

demo:
	uv run drift demo

test:
	uv run pytest

eval:
	uv run pytest tests/test_demo_golden.py tests/test_report_schema.py \
		-o addopts="" -v -p no:cacheprovider

test-live:
	DRIFT_LIVE_OLLAMA=1 uv run pytest -m live -p no:cacheprovider

check:
	uv run ruff check coursedrift tests
	uv run ruff format --check coursedrift tests
	uv run mypy coursedrift

fmt:
	uv run ruff format coursedrift tests
	uv run ruff check --fix coursedrift tests
