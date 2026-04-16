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

## 2. 多数据集训练（xhum-train-multi）

独立于 `lerobot-train`，支持将多个 LeRobot 数据集合并训练。

### 前提条件

- 已安装 xhum 工具链（`pip install -e .`）
- 各数据集已通过 `xhum-convert` 转换为 LeRobot V3 格式
- **所有数据集的 action / state 维度必须一致**（不一致会报错）

### 使用方式

```bash
xhum-train-multi --config path/to/multi_train.json
```

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
| `policy.type` | 策略类型，如 `act`、`diffusion`、`vqbet` 等 |
| `policy.path` | 预训练模型路径，`null` 表示从零开始 |
| `training.output_dir` | 模型输出目录 |
| `training.steps` | 总训练步数 |
| `training.save_freq` | 每隔多少步保存 checkpoint |
| `training.log_freq` | 每隔多少步打印日志 |
| `training.resume` | 设为 `true` 可从 checkpoint 恢复 |
| `training.resume_dir` | 恢复时指定 checkpoint 目录 |
| `use_imagenet_stats` | 是否用 ImageNet 统计量归一化图像 |
| `video_backend` | 视频解码后端，通常 `pyav` |

### 行为说明

- 自动取所有数据集的**特征交集**：不同的 camera key 会被自动禁用并警告
- 维度校验：所有公共特征（action、state 等）**必须维度一致**，不一致会报错
- Stats 聚合：对公共特征的统计量（mean / std 等）跨数据集聚合
- Checkpoint 保存在 `output_dir/checkpoints/` 下，格式与 `lerobot-train` 兼容
