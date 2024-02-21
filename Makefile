.PHONY: help install clean docs test ci mypy pyright pyupgrade isort black ruff release example-ci

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  clean      to clean the directory tree"
	@echo "  docs       to generate the documentation"
	@echo "  ci 	    to run the CI workflows"
	@echo "  mypy       to run the mypy static type checker"
	@echo "  pyright    to run the pyright static type checker"
	@echo "  pyupgrade  to run pyupgrade"
	@echo "  isort      to run isort"
	@echo "  black      to run black"
	@echo "  ruff 	    to run ruff"
	@echo "  test       to run the tests"
	@echo "  release    to perform all actions required for a release"
	@echo "  example-ci to run the CI workflows for the example scripts"

install:
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf cv2ext/*.egg-info
	rm -rf src/cv2ext/*.egg-info
	pyclean .
	rm -rf .mypy_cache
	rm -rf .ruff_cache

docs:
	python3 ci/build_example_docs.py
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/cv2ext/
	cd docs && make html

ci: pyupgrade ruff mypy isort black

mypy:
	python3 -m mypy src/cv2ext --config-file=pyproject.toml

pyright:
	python3 -m pyright --project=pyproject.toml

pyupgrade:
	-./ci/pyupgrade.sh

isort:
	python3 -m isort src/cv2ext

black:
	python3 -m black src/cv2ext --safe

ruff:
	python3 -m ruff ./src/cv2ext --fix --preview

test:
	./ci/run_tests.sh

example-ci: pyupgrade
	python3 -m ruff ./examples --fix --preview --ignore=T201,INP001,F841
	python3 -m isort examples
	python3 -m black examples --safe

release: clean ci test docs example-ci
