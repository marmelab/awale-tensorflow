install:
	$(workon cv)

run:
	python3 launcher.py $(COMMAND_ARGS)

lint:
	pep8

# Utility commands used to pass some arguments (COMMAND_ARGS)
SUPPORTED_COMMANDS := run
SUPPORTS_MAKE_ARGS := $(findstring $(firstword $(MAKECMDGOALS)), $(SUPPORTED_COMMANDS))
ifneq "$(SUPPORTS_MAKE_ARGS)" ""
  COMMAND_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  COMMAND_ARGS := $(subst :,\:,$(COMMAND_ARGS))
  $(eval $(COMMAND_ARGS):;@:)
endif
