.PHONY: setup fmt lint test run

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pre-commit install

fmt:
	black . && isort .

lint:
	flake8 .

test:
	pytest -q

run:
	python cli.py --video examples/synthetic.mp4 --out runs/example_run
