.PHONY: install test eval lint run-web run-api tf-validate docker-build clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
ADK := .venv/bin/adk

install:
	uv venv .venv --python 3.12
	uv pip install --python .venv/bin/python -e ".[dev]"

test:
	$(PYTEST) -v tests/

eval:
	$(PYTHON) eval/run_eval.py

lint:
	ruff check medchem_agent tests eval

run-web:
	$(ADK) web .

run-api:
	$(PYTHON) -m medchem_agent.main

tf-validate:
	terraform -chdir=terraform fmt -check -recursive
	terraform -chdir=terraform init -backend=false
	terraform -chdir=terraform validate

docker-build:
	docker build -t medchem-geap-agent:local .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
