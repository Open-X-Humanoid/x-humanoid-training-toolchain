# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running xhum modules (no `pip install`)

`xhum` is **not** pip-installed. `make install` / `make install-all` install **LeRobot only** (from the `lerobot/` submodule, pinned to v0.5.1). Run every `xhum.*` module from the repo root via:

```bash
./scripts/xhum-run xhum.<module> [args...]          # injects PYTHONPATH=src
# equivalent: PYTHONPATH=src python -m xhum.<module> ...
```

`./scripts/xhum-run` only injects `PYTHONPATH` and `exec`s `python -m`; there is no wrapper logic.

If the submodule is empty (fresh clone without `--recurse-submodules`), run `make install-lerobot` before anything else.

## Common commands

| Task | Command |
|------|---------|
| Install LeRobot submodule | `make install` (= `make install-lerobot`) |
| Install dev tools (pre-commit, pytest, ruff) | `make install-dev` |
| Update LeRobot submodule | `make update-lerobot && make install-lerobot` |
| Inspect an HDF5 trajectory | `python src/xhum/convert/inspect_h5.py <path>` |
| Convert HDF5 → LeRobot V3 | `./scripts/xhum-run xhum.convert.hdf5_to_lerobot --config ... --repo_id ... --src_root ... --tgt_path ...` |
| Train (single dataset, LeRobot native) | `lerobot-train --config_path=src/xhum/train/configs/act_tienkung.json` |
| Train (multi-dataset, xhum) | `./scripts/xhum-run xhum.train.train_multi --config <json>` |
| Monolithic ROS2 deploy | `./scripts/xhum-run xhum.deploy.ros2_deploy --config src/xhum/deploy/config.yaml` |
| Decoupled deploy — policy server (Py 3.12) | `cd src/xhum/deploy_decouple/policy && python policy_server.py --model_path ... --bind tcp://127.0.0.1:5555` |
| Decoupled deploy — ROS entry (Py 3.10) | `cd src/xhum/deploy_decouple/robot && python3 run.py --config ./my_robot.yaml` |
| Lint | `ruff check .` (`pyproject.toml`, target `py312`, line length 110, ignores `E501`) |

No `xhum`-level test suite is checked in yet; `pytest` runs only LeRobot's own tests from within `lerobot/`.

## Big-picture architecture

The repo is a thin toolchain on top of LeRobot:

```
lerobot/              # git submodule — huggingface/lerobot v0.5.1 (pip-installed via make install)
src/xhum/             # fully decoupled toolchain; NOT pip-installed
  convert/            # HDF5 → LeRobot V3 dataset (configs/*.json drive feature mapping + slicing)
  train/              # single-dataset wrapper + multi-dataset training (feature intersection + stats aggregation)
  deploy/             # monolithic ROS2 node (one Python env; rclpy + torch together)
  deploy_decouple/    # split-env deployment (Py3.12 policy ↔ Py3.10 ROS) over ZMQ
```

### Convert stage

`xhum.convert.hdf5_to_lerobot` maps HDF5 keys → LeRobot V3 features via a JSON config (`src/xhum/convert/configs/*.json`). Supports single-key, single-key + `slice`, multi-key concatenation (`hdf5_keys` + `slices`, `null` meaning full column range), and JPEG/PNG image `decode` with optional `resize`. `stats_override` in the config is only honored when `--stats-override` is passed on the CLI (incremental overlay — only declared fields replace auto-computed stats). Output follows LeRobot V3 layout (meta/ + data/chunk-* + videos/chunk-*), with multiple episodes packed into the same chunk and distinguished by per-episode timestamp ranges — this is normal.

### Train stage

- `lerobot-train` is the primary path (single dataset, standard LeRobot config).
- `xhum.train.train_multi` merges multiple LeRobot V3 datasets for joint training. It auto-intersects camera features (disabling non-shared cams with a warning) and **requires identical action/state dims across datasets**; mismatched dims fail loudly. Stats (mean/std/min/max) are aggregated across datasets for the common features. Checkpoints land in `output_dir/checkpoints/` in `lerobot-train`-compatible format.

### Deploy — two parallel implementations

There are **two** ROS2 deployment code paths that must stay behaviourally aligned:

1. **Monolithic** — `src/xhum/deploy/ros2_deploy.py` + `policy_agent.py`. Single Python env with both `rclpy` and `torch`. `hand_type` (inspire/brainco) and `mode` (model/replay) in one YAML.
2. **Decoupled** — `src/xhum/deploy_decouple/`. Process boundary = Python env boundary:

   | Dir | Python | Role |
   |-----|--------|------|
   | `policy/` | **3.12+** (torch + LeRobot) | `policy_server.py` (ZMQ REP) + `policy_agent.py` |
   | `robot/`  | **3.10** (rclpy)            | `run.py` entry → `ros2_node.py` / `replay_debug.py`; `policy_client.py` (ZMQ REQ); YAML + HDF5 replay loaders |
   | `wire/`   | **shared**, numpy + stdlib only | `obs_codec.py` (multipart protocol + `op: infer/reset`) + `trace_io.py` (PNG / joint dumps) |

   Control loop lives in ROS (client pulls one action per step); the policy process binds and replies. `robot/run.py` peeks `mode` from YAML and dispatches: `replay_debug` → `replay_debug.run` (no `rclpy` import, works on a laptop without ROS sourced); everything else → `ros2_node.main`. YAML `mode` is one of `model` / `replay` / `replay_actions` / `replay_debug`, where `replay_actions` is fully open-loop (no ZMQ, no policy server, no camera subscriptions). Legacy `mode=replay` + `replay_via_zmq: false` is normalized to `replay_actions` with a deprecation warning.

**Invariants to preserve when editing either deploy path:**
- `deploy_decouple/policy/policy_agent.py` must stay in sync with `deploy/policy_agent.py` (same obs dict shape: `images[<short_cam_name>]` + `arm_gripper_joints`; same preprocessor/postprocessor wiring when `policy_preprocessor.json` / `policy_postprocessor.json` sit beside the checkpoint).
- `obs_camera_key` must equal the `<X>` in the checkpoint's `observation.images.<X>`. Defaults differ by hand type (`inspire` → `camera_head`, `brainco` → `camera`).
- **`model_path` is only a `policy_server --model_path` flag.** The robot YAML must not carry it — the ZMQ wire carries observations and actions only. Swap checkpoints by restarting the policy server.
- `wire/obs_codec.py` `PROTOCOL_VERSION` is a hard version boundary; bumping it requires redeploying both processes together. Only stdlib + numpy may be imported from `wire/`.
- Neither `deploy_decouple/policy/` nor `deploy_decouple/robot/` is pip-installed. Each entry script injects its siblings + `wire/` into `sys.path` at startup. Do not introduce cross-imports that assume a package install.
- `policy_zmq_timeout_ms` default is `120000` on purpose — first ACT inference can exceed 10 s (model load + warm-up). Don't shrink it casually.

### Gitignored paths under `deploy_decouple`

`src/xhum/deploy_decouple/robot/config/*.yaml` is gitignored except `config_zmq.example.yaml` and `replay_debug.yaml`. Local helper scripts under `src/xhum/deploy_decouple/scripts/` are gitignored except `README.md`, `stat_hdf5_firstframe_mean.py`, and `eval_policy_from_hdf5.py`. Runtime dumps (`deploy_decouple/tmp/`, any `debug_server_images/`, `debug_client_joint/`, `debug_server_joint/`) are ignored. Don't commit local YAMLs, image dumps, or joint-trace `.npy` files.

## Repo conventions

- Python target is **3.12** (`pyproject.toml`, `ruff target-version = py312`). The one exception is `src/xhum/deploy_decouple/robot/` and `deploy_decouple/wire/`, which must stay runnable under **Python 3.10** for ROS2 Humble. Avoid 3.11+ syntax there.
- Ruff config: line length 110, ignores `E501`, enables `E/W/F/I/B/C4/SIM`, `known-first-party = ["xhum", "lerobot"]`.
- Most user-facing docs come in an English + `_zh` Chinese pair. When adding a user-facing doc, update both or clearly mark the scope.

## Subdirectory conventions

- **`src/xhum/deploy_decouple/`** — prefer inline **WHY** comments for non-obvious invariants (env boundaries, protocol versioning, no-install `sys.path` injection, ZMQ timeouts, threading/executor caveats). This overrides the default "no comments" stance for this subtree only. Keep comments short (1–3 lines); do not narrate obvious lines.
