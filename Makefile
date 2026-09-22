PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= configs/default.yaml

.PHONY: setup venv install lock data lint format test clean clean-data

## Everything a clean clone needs before any other target will work.
setup: install
	$(MAKE) data

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(PIP) install --quiet --upgrade pip

## Install from the lock file, not from pyproject ranges, so the environment matches the
## one every number in the README was produced in. --no-deps on the project install stops
## pip re-resolving and quietly upgrading a pinned transitive dependency.
install: venv
	$(PIP) install --requirement requirements.lock
	$(PIP) install --no-deps --editable .

## Regenerate requirements.lock from the ranges in pyproject.toml. Run deliberately, not as
## part of setup: the point of a lock file is that it does not move on its own.
lock: venv
	$(PIP) install --editable ".[dev]"
	$(PIP) freeze --exclude-editable > requirements.lock

data:
	$(PY) scripts/download_data.py --config $(CONFIG)

lint:
	$(VENV)/bin/ruff check src scripts tests
	$(VENV)/bin/black --check src scripts tests

format:
	$(VENV)/bin/ruff check --fix src scripts tests
	$(VENV)/bin/black src scripts tests

test:
	$(PY) -m pytest -q -m "not integration"

## Removes reproducible outputs and caches only. The dataset is a 5GB download; deleting it
## needs its own target so a stray `make clean` cannot cost anyone an afternoon.
clean:
	rm -rf artifacts results .pytest_cache .ruff_cache
	find src scripts tests -name __pycache__ -type d -prune -exec rm -rf {} +

clean-data:
	rm -rf data

.PHONY: train evaluate serve reproduce integration
train:
	$(PY) scripts/train.py --config "$(CONFIG)"

evaluate:
	$(PY) scripts/evaluate.py --config "$(CONFIG)"

serve:
	$(PY) scripts/serve.py --config "$(CONFIG)"

integration:
	$(PY) -m pytest -q -m integration

## Recursive calls keep the pipeline sequential even under `make -j`.
reproduce:
	$(MAKE) setup
	$(MAKE) train
	$(MAKE) evaluate
	$(MAKE) benchmark
	$(MAKE) report
	$(MAKE) lint test
	$(MAKE) integration

.PHONY: benchmark guide
benchmark:
	$(PY) scripts/benchmark_api.py --config "$(CONFIG)"

guide:
	$(PY) -m http.server 8765 --bind 127.0.0.1 --directory .

.PHONY: report
report:
	$(PY) scripts/snapshot_report.py --config "$(CONFIG)"
