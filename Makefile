
SHELL = /bin/bash

VENV = venv-hnx
PYTHON_VENV = $(VENV)/bin/python3
PYTHON3 = python3


## Test

test: clean venv
	@$(PYTHON_VENV) -m pip install -e .'[auto-testing]' --use-pep517
	@$(PYTHON_VENV) -m tox -e py38 -e py311

test-ci:
	@$(PYTHON3) -m pip install -e .'[auto-testing]' --use-pep517
	@$(PYTHON3) -m pip install 'pytest-github-actions-annotate-failures>=0.1.7'
	pre-commit install
	pre-commit run --all-files
	@$(PYTHON3) -m tox -e py38 -r

.PHONY: test, test-ci

## Continuous Deployment
## Assumes that scripts are run on a container or test server VM

### Publish to PyPi
publish-deps:
	@$(PYTHON3) -m pip install -e .'[packaging]'

build-dist: publish-deps clean
	@$(PYTHON3) -m build --wheel --sdist
	@$(PYTHON3) -m twine check dist/*

## Assumes the following environment variables are set: TWINE_USERNAME, TWINE_PASSWORD, TWINE_REPOSITORY_URL,
## See https://twine.readthedocs.io/en/stable/#environment-variables
publish-to-pypi: publish-deps build-dist
	@echo "Publishing to PyPi"
	$(PYTHON3) -m twine upload dist/*

.PHONY: build-dist publish-to-test-pypi publish-to-pypi publish-deps

### Update version

version-deps: clean-venv venv
	@$(PYTHON_VENV) -m pip install .'[releases]'

bump-version-major: version-deps
	bump2version --dry-run --verbose major
	bump2version --verbose major

bump-version-minor: version-deps
	bump2version --dry-run --verbose minor
	bump2version --verbose minor

bump-version-patch: version-deps
	bump2version --dry-run --verbose patch
	bump2version --verbose patch

.PHONY: version-deps bump-version-major bump-version-minor bump-version-patch

#### Documentation

docs-deps:
	@$(PYTHON3) -m pip install -e .'[documentation]' --use-pep517

commit-docs:
	git add -A
	git commit -m "Bump version in docs"

.PHONY: commit-docs docs-deps

## Environment

clean-venv:
	rm -rf $(VENV)

clean: clean-venv
	rm -rf .out .pytest_cache .tox *.egg-info dist build

venv:
	@$(PYTHON3) -m venv $(VENV);

all-deps:
	@$(PYTHON3) -m pip install -e .'[all]' --use-pep517

.PHONY: venv

.PHONY: clean clean-venv venv all-deps
