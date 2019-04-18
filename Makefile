ENVRUN = poetry run
PACKAGE = alchina
LINE_LENGTH = 88

lint:
	@$(ENVRUN) flake8 $(PACKAGE) tests --max-line-length $(LINE_LENGTH)

type:
	@$(ENVRUN) mypy $(PACKAGE) --ignore-missing-imports

test:
	@$(ENVRUN) py.test --cov=$(PACKAGE) --cov-report term-missing -vs

format:
	@$(ENVRUN) black -l $(LINE_LENGTH) -S $(PACKAGE) tests
