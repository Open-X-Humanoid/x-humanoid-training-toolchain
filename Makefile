.PHONY: install install-all install-lerobot install-xhum install-dev update-lerobot

# Default: LeRobot only. The ``src/xhum`` package is optional — use ``install-all`` when you need ``xhum-convert`` etc.
install: install-lerobot

# LeRobot submodule + editable xhum (console scripts: xhum-convert, …)
install-all: install-lerobot install-xhum

# Step 1: Install lerobot from submodule
install-lerobot:
	git submodule update --init --recursive
	pip install -e ./lerobot

# Step 2: Install xhum toolchain (optional)
install-xhum:
	pip install -e .

# Development install: full toolchain + dev extras
install-dev: install-all
	pip install -e ".[dev]"

# Update lerobot submodule to latest upstream
update-lerobot:
	cd lerobot && git fetch origin && git checkout main && git pull origin main
	@echo "LeRobot updated. Run 'make install-lerobot' to reinstall."
