# Native Training

This folder provides a native LeRobot training launcher script.

## Files

- `run_train_native.sh`: runs `lerobot-train` with a verified ACT command template.

## Prerequisites

- `lerobot-train` command is available in your current environment.
- `HF_LEROBOT_HOME` is set to your converted dataset root.

Example:

```bash
export HF_LEROBOT_HOME=/media/jushen/neil-liu/dataNmodels/lerobot_dataset
```

## Run

From project root:

```bash
bash src/xhum/train/run_train_native.sh
```

## Pass extra native args

Additional args are passed to `lerobot-train`:

```bash
bash src/xhum/train/run_train_native.sh --resume=true
```

## Customize training values

Edit values directly in `run_train_native.sh`:

- `--dataset.repo_id`
- `--output_dir`
- `--job_name`
- `--batch_size`
- `--steps`
- `--policy.push_to_hub`
