PYTHON ?= python3
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python3
VENV_PIP := $(VENV)/bin/pip
SCRIPT := signal_grapher.py

.PHONY: all install run clean help

all: run

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "==> Creating virtual environment in $(VENV)..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@echo "==> Installing dependencies (numpy, matplotlib)..."
	$(VENV_PIP) install numpy matplotlib
	@echo "==> Installation complete."

$(VENV_PYTHON):
	@$(MAKE) install

run: $(VENV_PYTHON)
	@$(VENV_PYTHON) $(SCRIPT) $(ARGS)

clean:
	@echo "==> Removing virtual environment and cache files..."
	rm -rf $(VENV) __pycache__ *.pyc

help:
	@echo "Available targets:"
	@echo "  make install            Create .venv and install numpy & matplotlib"
	@echo "  make run                Launch the real-time Signal Grapher"
	@echo "  make run ARGS='\"3,4\"'   Launch with initial commands"
	@echo "  make clean              Remove .venv and Python cache files"
