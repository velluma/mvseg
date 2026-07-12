# Convenience commands. All Python runs through `uv run` for reproducibility.
.PHONY: help setup lock lint fmt test train eval predict splits inspect docker-build docker-shell clean

help:
	@echo "make setup        - uv sync (create env from lock file, with dev extras)"
	@echo "make lock         - regenerate uv.lock"
	@echo "make lint         - ruff check + format check"
	@echo "make fmt          - ruff format (apply)"
	@echo "make test         - run pytest (CPU smoke tests)"
	@echo "make train        - run training (override with ARGS='experiment=resunet_baseline')"
	@echo "make eval         - evaluate a checkpoint (ARGS='ckpt_path=...')"
	@echo "make predict      - run inference (ARGS='ckpt_path=... input=...')"
	@echo "make splits       - generate data/splits/splits.json"
	@echo "make inspect      - sanity-check the dataset"
	@echo "make docker-build - build the GPU docker image"
	@echo "make docker-shell - open a shell in the GPU container"

setup:
	uv sync --extra dev

lock:
	uv lock

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -m "not gpu"

train:
	uv run python -m mvseg.train $(ARGS)

eval:
	uv run python -m mvseg.evaluate $(ARGS)

predict:
	uv run python -m mvseg.predict $(ARGS)

splits:
	uv run python scripts/prepare_splits.py $(ARGS)

inspect:
	uv run python scripts/inspect_data.py $(ARGS)

docker-build:
	docker compose build

docker-shell:
	docker compose run --rm mvseg bash

clean:
	rm -rf outputs multirun wandb lightning_logs .pytest_cache .ruff_cache
