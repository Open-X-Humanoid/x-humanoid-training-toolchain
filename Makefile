.PHONY: install install-lerobot install-xhum install-dev update-lerobot

# Install everything (lerobot submodule + xhum toolchain)
install: install-lerobot install-xhum

# Step 1: Install lerobot from submodule
install-lerobot:
	git submodule update --init --recursive
	pip install -e ./lerobot

# Step 2: Install xhum toolchain
install-xhum:
	pip install -e .

# Development install with extra tools
install-dev: install
	pip install -e ".[dev]"

# Update lerobot submodule to latest upstream
update-lerobot:
	cd lerobot && git fetch origin && git checkout main && git pull origin main
	@echo "LeRobot updated. Run 'make install-lerobot' to reinstall."
