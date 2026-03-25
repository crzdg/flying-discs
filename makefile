.PHONY: mypy lint test all

all: style mypy lint test

mypy:
	@hatch run dev:typing

lint:
	@hatch run dev:lint

test:
	@hatch run dev:no-cov

style:
	@hatch run style:check

prune-hatch:
	@hatch env prune

update: pip-compile

pip-compile:
	@hatch run tools:update -U -o requirements/main.txt --annotation-style=line --resolver=backtracking
	@hatch run tools:update -U --extra dev --extra test --extra lint --extra type --extra style -o requirements/dev.txt --annotation-style=line --resolver=backtracking

release-%:
	@hatch version $*
