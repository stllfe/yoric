# Common developer commands

SHELL = /bin/bash
VENV = venv
PKGS = $(VENV)/bin
DIRS = yogurt scripts tests

DIFF := 1
COMM := 0
ifeq ($(DIFF), 1)
    PYFILES := $(shell git diff --name-only $(if $(filter 1, $(COMM)), --cached) --diff-filter=ACMRTUXB HEAD | grep -E '\.py$$')
else
    PYFILES := $(shell find $(DIRS) -name '*.py' -type f -print | tr '\n' ' ')
endif

.PHONY: help
help:
	@echo 'Usage: '
	@echo '  make <command> [DIRS=dir1 dir2 ... DIFF={0,1} COMM={0, 1}]'
	@echo
	@echo 'Options:'
	@echo '  DIFF   whether to run commands only on changed files (default: $(DIFF))'
	@echo '  COMM   whether to use diff on staged files only (default: $(COMM))'
	@echo '  DIRS   directories to use with commands if DIFF=0'
	@echo
	@echo 'Commands:'
	@echo '  venv   create a virtual environment'
	@echo '  lint   check code quality'
	@echo '  types  check static types'
	@echo '  style  run code formatting'
	@echo '  clean  clean all unnecessary files'
	@echo '  tests  run tests'
	@echo
	@echo '  opt    show resolved option values'
	@echo '  all    run all above (except venv)'

.PHONY: opt
opt:
	@echo 'SHELL=$(SHELL)'
	@echo 'VENV=$(VENV)'
	@echo 'PKGS=$(PKGS)'
	@echo 'DIFF=$(DIFF)'
	@echo 'COMM=$(COMM)'
	@echo 'DIRS=$(DIRS)'
	@echo 'PYFILES=$(PYFILES)'

.show:
	@echo 'Input:'
	@echo '$(PYFILES)' | tr ' ' '\n' | sed 's/^/  /'
	@echo

.PHONY: types
types: .show
	@$(PKGS)/mypy --install-types --non-interactive $(PYFILES)

.PHONY: lint
lint: .show
	@source $(PKGS)/activate && \
	echo 'Flake8:'; \
	flake8 $(PYFILES) && echo 'OK ✓'; \
	echo; \
	echo 'Pylint:'; \
	pylint -rn $(PYFILES) && echo 'OK ✓'

.PHONY: style
style: .show
	@$(PKGS)/isort $(PYFILES)
	@$(PKGS)/autoflake8 --exit-zero-even-if-changed -r --in-place $(PYFILES)
	@$(PKGS)/pyupgrade --exit-zero-even-if-changed --keep-runtime-typing --py39-plus $(PYFILES)
	@$(PKGS)/unify -r --in-place $(PYFILES)
	@$(PKGS)/black $(PYFILES)

.ONESHELL:
venv:
	@python -m venv $(VENV) && source $(PKGS)/activate
	@pip install --upgrade pip setuptools wheel
	@pip install -e .
	@pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
	@pytest --fixtures --collect-only &> /dev/null
	@echo '✓ Python virtual environment initialized sucessfully!'

.PHONY: clean
clean:
	@find . -type f -name '*.DS_Store' -ls -delete
	@find . | grep -E '(__pycache__|\.pyc|\.pyo)' | xargs rm -rf
	@find . | grep -E '.pytest_cache' | xargs rm -rf
	@find . | grep -E '.ipynb_checkpoints' | xargs rm -rf
	@find . | grep -E '.trash' | xargs rm -rf
	@rm -f .coverage
	@echo '✓ Project root cleaned!'

.PHONY: tests
tests:
	@$(PKGS)/pytest

all: style types lint tests clean
