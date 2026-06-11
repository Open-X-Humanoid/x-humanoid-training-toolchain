# Training

## 1. 单数据集训练（lerobot-train）

使用 LeRobot 原生 `lerobot-train` 命令。

```bash
export HF_LEROBOT_HOME=/media/jushen/neil-liu/dataNmodels/lerobot_dataset
bash src/xhum/train/run_train_native.sh
```

可在 `run_train_native.sh` 中修改 `--dataset.repo_id`、`--output_dir` 等参数。
追加参数直接附在命令后：

```bash
bash src/xhum/train/run_train_native.sh --resume=true
```

---

## 2. 多数据集训练（`train_multi`）

独立于 `lerobot-train`，支持将多个 LeRobot 数据集合并训练。

### 前提条件

- 单卡：在仓库根目录用 **`./scripts/xhum-run`**（或 `PYTHONPATH=src`）运行 `xhum.train.train_multi`
- 多卡：用 **`accelerate launch --module xhum.train.train_multi`**（见下；不能对 `./scripts/xhum-run` 做 accelerate launch，它是 bash 脚本）
- 各数据集已转换为 LeRobot V3 格式（`./scripts/xhum-run xhum.convert.hdf5_to_lerobot ...`）
- **所有数据集的 action / state 维度必须一致**（不一致会报错）

### 使用方式

单卡：

```bash
./scripts/xhum-run xhum.train.train_multi --config path/to/multi_train.json
```

多卡（示例 8 GPU）：

```bash
export PYTHONPATH=src
export PYTORCH_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --num_processes 8 \
  --module xhum.train.train_multi \
  -- --config path/to/multi_train.json
```

`batch_size` 为**每张 GPU** 的 batch；有效 batch = `batch_size × GPU 数`。

### 配置文件格式

参考 `configs/multi_train_example.json`：

```json
{
  "datasets": [
    {
      "repo_id": "dataset_a",
      "root": "/absolute/path/to/dataset_a",
      "episodes": null
    },
    {
      "repo_id": "dataset_b",
      "root": "/absolute/path/to/dataset_b",
      "episodes": null
    }
  ],
  "policy": {
    "type": "act",
    "path": null,
    "push_to_hub": false
  },
  "training": {
    "output_dir": "/path/to/output",
    "job_name": "multi_act_run_001",
    "batch_size": 32,
    "steps": 200000,
    "num_workers": 4,
    "save_freq": 20000,
    "log_freq": 200,
    "seed": 1000
  },
  "use_imagenet_stats": true,
  "video_backend": "pyav"
}
```

### 配置项说明

| 字段 | 说明 |
|------|------|
| `datasets[].repo_id` | 数据集标识（自定义名称即可） |
| `datasets[].root` | 数据集绝对路径（包含 `meta/`、`data/`、`videos/`） |
| `datasets[].episodes` | 指定使用的 episode 索引列表，`null` 表示全部 |
| `rename_map` | 可选；顶层或 `policy` 内均可。数据集 feature key → 策略 key，用于相机名对齐（见 §3） |
| `policy.type` | 策略类型，如 `act`、`diffusion`、`vqbet`、`pi05` 等 |
| `policy.path` | 预训练模型路径，`null` 表示从零开始；π₀.₅ 填 `lerobot/pi05_base` |
| `policy.normalization_mapping` | 可选；π₀.₅ 用 `MEAN_STD` 时无需 quantile stats（见 §3） |
| `policy.gradient_checkpointing` 等 | 可选；任意策略 dataclass 已有字段均可覆盖（如 `dtype`、`freeze_vision_encoder`） |
| `training.output_dir` | 模型输出目录 |
| `training.steps` | 总训练步数 |
| `training.save_freq` | 每隔多少步保存 checkpoint |
| `training.log_freq` | 每隔多少步打印日志 |
| `training.resume` | 设为 `true` 可从 checkpoint 恢复 |
| `training.resume_dir` | 恢复时指定 checkpoint 目录 |
| `training.job_name` | WandB **run 名**（`wandb.run_name` 为空时用它） |
| `use_imagenet_stats` | 是否用 ImageNet 统计量归一化图像 |
| `video_backend` | 视频解码后端，通常 `pyav` |
| `wandb.enable` | 是否启用 WandB（仅主进程） |
| `wandb.project` | WandB **项目名**（必填，启用时） |
| `wandb.entity` | WandB 团队/用户，如 `714305606-peking-university` |
| `wandb.run_name` | 可选；覆盖 run 显示名，默认 `training.job_name` |
| `wandb.group` | 可选；run 分组，默认 `policy:<type>-seed:<seed>` |
| `wandb.run_id` | 可选；恢复训练时指定已有 run id（`resume=true` 时也可从 output_dir 自动读取） |
| `wandb.mode` | `online` / `offline` / `disabled` |
| `wandb.disable_artifact` | `true` 时不把 checkpoint 上传为 artifact |
| `wandb.add_tags` | 默认 `true`；自动打 `policy:` / `seed:` / `dataset:` 标签 |
| `wandb.notes` | 可选；run 备注 |

### 行为说明

- 自动取所有数据集的**特征交集**（`rename_map` 重命名后再求交）：不同的 camera key 会被自动禁用并警告
- 维度校验：所有公共特征（action、state 等）**必须维度一致**，不一致会报错
- Stats 聚合：对公共特征的统计量（mean / std 等）跨数据集聚合；有 `rename_map` 时 stats key 同步重命名
- Checkpoint 保存在 `output_dir/checkpoints/` 下，格式与 `lerobot-train` 兼容；同时写入 `pretrained_model/multi_train_config.json` 备份完整训练 config
- 分布式保存：所有 rank 在 `save_freq` 步同步等待，仅 rank 0 写盘
- WandB（`wandb_multi.py`）：每 `log_freq` 步记录 loss/lr 等到 `train/*`；每 `save_freq` 步可上传 model artifact（需 `pip install wandb` 且 `WANDB_API_KEY` 已配置）

---

## 3. 多数据集 π₀.₅（MEAN_STD 归一化）

π₀.₅ 默认使用 **QUANTILES**；若转换后的 dataset 只有 `mean`/`std`（无 `q01`…`q99`），在 config 里显式指定 **MEAN_STD**：

```json
"policy": {
  "type": "pi05",
  "path": "lerobot/pi05_base",
  "normalization_mapping": {
    "ACTION": "MEAN_STD",
    "STATE": "MEAN_STD",
    "VISUAL": "IDENTITY"
  },
  "gradient_checkpointing": true,
  "dtype": "bfloat16",
  "compile_model": false,
  "freeze_vision_encoder": false,
  "train_expert_only": false
}
```

**无需**运行 `augment_dataset_quantile_stats.py`。

| 字段 | 说明 |
|------|------|
| `freeze_vision_encoder` | `true` 冻结视觉编码器，仅微调 action expert |
| `train_expert_only` | 仅训练 expert 分支 |
| `compile_model` | 是否 `torch.compile` 模型 |
| `gradient_checkpointing` | 降低显存占用，π₀.₅ 微调建议开启 |

π₀.₅ 预训练 (`pi05_base`) 期望相机 key 为 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`。转换数据集若使用 `camera_head` / `camera_left` / `camera_right`，在 config 顶层加 `rename_map`（数据集 key → 策略 key）：

```json
"rename_map": {
  "observation.images.camera_head": "observation.images.base_0_rgb",
  "observation.images.camera_left": "observation.images.left_wrist_0_rgb",
  "observation.images.camera_right": "observation.images.right_wrist_0_rgb"
}
```

`rename_map` 会作用于：数据集样本读取、meta 特征名、stats 归一化、以及 preprocessor 中的 `RenameObservationsProcessorStep`。

π₀.₅ 的 preprocessor/postprocessor 由代码构建（`make_pi05_pre_post_processors`），不直接加载 `pi05_base` 附带的 `policy_preprocessor.json`，避免与 vendored lerobot 版本不一致。

### 完整 config 示例

参考仓库内 `configs/multi_train_pi05_tianshu_73_new_white_upward_downward_unfreeze_ve_0611.json`（2 个 dataset + WandB）或 `configs/multi_train_pi05_tianshu_72_all_white_black_upward_downward_unfreeze_vision_encoder_0607.json`（4 个 task 全量）。

```json
{
  "datasets": [ ... ],
  "rename_map": { ... },
  "policy": { "type": "pi05", "path": "lerobot/pi05_base", ... },
  "training": { "output_dir": "...", "job_name": "...", "batch_size": 64, ... },
  "wandb": {
    "enable": true,
    "project": "pi05-tianshu-demo",
    "entity": "your-wandb-entity",
    "run_name": "my_run_name"
  }
}
```

### 运行

```bash
cd lerobot && pip install -e ".[pi]" && cd ..
# 8 GPU（编辑脚本内 CONFIG 指向你的 json）
bash src/xhum/train/run_train_pi05_multi.example_8gpu_fast.sh

# 或换 config 而不改脚本
CONFIG=src/xhum/train/configs/multi_train_pi05_tianshu_73_new_white_upward_downward_unfreeze_ve_0611.json \
  bash src/xhum/train/run_train_pi05_multi.example_8gpu_fast.sh
```

单数据集 π₀.₅ 仍可用 `lerobot-train`，在 CLI 或 JSON config 中设置同上 `normalization_mapping` 与 `rename_map`（若相机 key 需对齐）。

---

## 辅助脚本与示例 Config

**版本库内**（可直接参考或复制修改）：

| 文件 | 说明 |
|------|------|
| `configs/multi_train_example.json` | ACT 双数据集最小示例 |
| `configs/multi_train_pi05_tianshu_73_new_white_upward_downward_unfreeze_ve_0611.json` | π₀.₅ 双 dataset + WandB 示例 |
| `configs/multi_train_pi05_tianshu_72_all_white_black_upward_downward_unfreeze_vision_encoder_0607.json` | π₀.₅ 四 task 全量示例 |
| `run_train_pi05_multi.example_8gpu_fast.sh` | 8 GPU 启动模板（含 `PYTORCH_ALLOC_CONF`） |
| `run_train_native.example.sh` | 单数据集 `lerobot-train` 模板 |
