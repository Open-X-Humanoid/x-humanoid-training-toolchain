.PHONY: install install-all install-lerobot install-dev update-lerobot

# Default: LeRobot only. Run xhum from source: ./scripts/xhum-run xhum.<module> …
install: install-lerobot

# Same as ``install`` (LeRobot only).
install-all: install-lerobot
	@echo "xhum (no pip install): ./scripts/xhum-run xhum.convert.hdf5_to_lerobot --help"

# Step 1: Install lerobot from submodule
install-lerobot:
	git submodule update --init --recursive
	pip install -e ./lerobot

# LeRobot + dev CLI tools only (use ./scripts/xhum-run for xhum modules)
install-dev: install-lerobot
	pip install "pre-commit>=3.7.0" "pytest>=8.1.0" "ruff>=0.4.0"

# Update lerobot submodule to latest upstream
update-lerobot:
	cd lerobot && git fetch origin && git checkout main && git pull origin main
	@echo "LeRobot updated. Run 'make install-lerobot' to reinstall."
