PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= configs/default.yaml

.PHONY: setup venv install lock data lint format test clean clean-data

## Everything a clean clone needs before any other target will work.
setup: install data

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
	$(PY) -m pytest -q

## Removes reproducible outputs and caches only. The dataset is a 5GB download; deleting it
## needs its own target so a stray `make clean` cannot cost anyone an afternoon.
clean:
	rm -rf artifacts results .pytest_cache .ruff_cache
	find src scripts tests -name __pycache__ -type d -prune -exec rm -rf {} +

clean-data:
	rm -rf data
