# HDF5 → LeRobot V3 数据转换

将 HDF5 采集数据转换为 LeRobot V3 Dataset 格式，供后续训练使用。

## 整体流程

```
Step 1: inspect_h5.py  →  查看 HDF5 里有哪些 key / shape
Step 2: 编辑 config JSON →  决定哪些 key 映射到 state、action、image
Step 3: hdf5_to_lerobot（仓库根目录 ``./scripts/xhum-run xhum.convert.hdf5_to_lerobot``）→  执行转换
```

---

## Step 1: 查看 HDF5 数据结构

在编辑 config 之前，先用 `inspect_h5.py` 看清楚原始 HDF5 里有哪些 key：

```bash
# 指向 src_root 目录，自动找第一个 episode 的 trajectory.hdf5
python src/xhum/convert/inspect_h5.py /path/to/success_episodes

# 或直接指向某个 HDF5 文件
python src/xhum/convert/inspect_h5.py /path/to/episode/data/trajectory.hdf5

# 显示更多样本行
python src/xhum/convert/inspect_h5.py /path/to/success_episodes --rows 5

# 指定 episode 内 HDF5 相对路径（默认 data/trajectory.hdf5）
python src/xhum/convert/inspect_h5.py /path/to/success_episodes --episode_path trajectory.hdf5
```

输出示例：

```
── Numeric datasets (candidates for state / action) ──

  Key                                    Shape        Dtype    Dim
  ────────────────────────────────────── ──────────── ──────── ────
  master/joint_position                  (393, 16)    float64  16
  puppet/joint_position                  (393, 26)    float32  26

  >> master/joint_position  (showing first 2 rows)
     row 0: [0]0.0368  [1]0.0873  [2]-0.2439  ...
     row 1: [0]0.0353  [1]0.0873  [2]-0.2439  ...

── Image / compressed datasets ──

  observations/rgb_images/camera_camera  393          object
  observations/depth_images/camera_camera 393         object
```

可选：自动生成一份 config 模板：

```bash
python src/xhum/convert/inspect_h5.py /path/to/episodes --gen_config configs/my_new_config.json
```

---

## Step 2: 编辑 Config JSON

Config 文件在 `src/xhum/convert/configs/` 目录下。文件结构：

```jsonc
{
    "dataset": {
        "fps": 30,
        "robot_type": "tienkung"
    },
    // episode 子目录内 HDF5 相对路径；常见两种布局：
    //   data/trajectory.hdf5  — 嵌套在 data/ 下（如 dvt217）
    //   trajectory.hdf5       — 直接在 episode 根目录（如 dvt228 / evt2-17）
    // 转换时若该路径不存在，会自动回退尝试 episode 根目录的 trajectory.hdf5
    "episode_path": "data/trajectory.hdf5",

    // features: 定义输出 dataset 的特征名、类型、维度
    "features": {
        "observation.state": { "dtype": "float32", "shape": [26], "names": null },
        "action":            { "dtype": "float32", "shape": [26], "names": null },
        "observation.images.camera": {
            "dtype": "video", "shape": [360, 640, 3],
            "names": ["height", "width", "channels"]
        }
    },

    // mappings: HDF5 key → feature 的映射关系
    "mappings": [ ... ],

    // stats_override (可选): 手动指定 stats，需配合 --stats-override 开关
    "stats_override": { ... }
}
```

### Mapping 写法

**单 Key 映射**（最常用）：

```json
{
    "hdf5_key": "puppet/joint_position",
    "feature_key": "observation.state"
}
```

**单 Key + 列切片**（只取部分列）：

```json
{
    "hdf5_key": "puppet/joint_position",
    "slice": [0, 16],
    "feature_key": "action"
}
```

**数值缩放**（读取后除以标量，常用于原始值需归一化的字段）：

```json
{
    "hdf5_key": "puppet/end_effector_left_position_align/data",
    "divide_by": 100.0,
    "feature_key": "observation.state"
}
```

**多 Key 拼接**（沿 axis-1 拼接多个 HDF5 dataset）：

```json
{
    "hdf5_keys": ["master/joint_position", "puppet/joint_position"],
    "slices": [[0, 8], [7, 13]],
    "feature_key": "action"
}
```

其中 `slices` 逐 key 对应，`null` 表示取全部列；`divide_by` 同样逐 key 对应（单 key 时也可写标量）：

```json
{
    "hdf5_keys": ["key_a", "key_b"],
    "slices": [null, [0, 6]],
    "divide_by": [null, 100.0],
    "feature_key": "action"
}
```

多 key 拼接 + 缩放完整示例（见 `configs/tianshu_72_express_demo_pi05_puppet.json`）：

```json
{
    "hdf5_keys": [
        "puppet/arm_left_position_align/data",
        "puppet/end_effector_left_position_align/data",
        "puppet/arm_right_position_align/data",
        "puppet/end_effector_right_position_align/data",
        "puppet/head_position_align/data"
    ],
    "feature_key": "observation.state",
    "divide_by": [null, 100.0, null, 100.0, null]
}
```

**图像映射**（需指定 decode 方式）：

```json
{
    "hdf5_key": "observations/rgb_images/camera_camera",
    "feature_key": "observation.images.camera",
    "decode": "jpeg",
    "resize": [640, 360]
}
```

### Stats Override（可选）

如果需要手动指定某个 feature 的归一化统计值（mean / std 等），在 config 中加 `stats_override` 段，并在运行时传 `--stats-override` 开关：

```json
"stats_override": {
    "action": {
        "mean": [0.036, 0.079, ...],
        "std":  [0.001, 0.0004, ...]
    }
}
```

只有显式传了 `--stats-override` 才会生效，否则忽略此段。
覆盖是增量的——只替换声明的字段，其余保留自动计算的值。

---

## Step 3: 执行转换

```bash
# 基本用法（在仓库根目录）
./scripts/xhum-run xhum.convert.hdf5_to_lerobot \
  --config src/xhum/convert/configs/dvt217_stack_cube.json \
  --repo_id dvt217_stack_cube \
  --src_root /path/to/success_episodes \
  --tgt_path /path/to/output/lerobot_dataset \
  --task_name stack_cube

# 启用手动 stats 覆盖
./scripts/xhum-run xhum.convert.hdf5_to_lerobot \
  --config src/xhum/convert/configs/dvt217_stack_cube.json \
  --repo_id dvt217_stack_cube \
  --src_root /path/to/success_episodes \
  --tgt_path /path/to/output/lerobot_dataset \
  --task_name stack_cube \
  --stats-override

# 控制图像解码线程数（默认自动）
./scripts/xhum-run xhum.convert.hdf5_to_lerobot --config ... --decode-workers 4
```

### 参数说明

| 参数 | 必填 | 说明 |
|------|------|------|
| `--config` | 是 | Config JSON 文件路径 |
| `--repo_id` | 是 | 输出 dataset 的 repo ID（即输出子目录名） |
| `--src_root` | 是 | HDF5 源数据根目录（下面是各 episode 子目录） |
| `--tgt_path` | 是 | 输出 dataset 的父目录 |
| `--task_name` | 否 | 任务名（默认 `default_task`） |
| `--decode-workers` | 否 | 图像解码线程数（0=自动，默认） |
| `--stats-override` | 否 | 启用 config 中的 stats_override（默认关闭） |

### 运行时行为

- **进度计数**：每个 episode 保存成功后输出 `[saved/total]` 进度（如 `[3/120] Saved episode: ...`）。
- **编码日志抑制**：默认设置 `SVT_LOG=1`（仅输出 SVT-AV1 错误）并压低 ffmpeg/libav 日志，避免大批量转换时刷屏。需要完整编码器诊断时可手动设置 `SVT_LOG=3`。
- **内存提示**：`--decode-workers` 控制每个 episode 内的 JPEG/PNG 解码线程数，不并行 LeRobot 写入；内存紧张时可设为 `1`。

### 输出结构

```
tgt_path/repo_id/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet
│       └── ...
├── meta/
│   ├── info.json        ← 数据集元信息（features / shape / fps）
│   ├── stats.json       ← 归一化统计值（mean / std / min / max）
│   └── tasks.parquet
└── videos/
    └── observation.images.camera/
        └── chunk-000/
            ├── file-000.mp4
            └── ...
```

---

## 完整示例

```bash
# 1. 查看数据
python src/xhum/convert/inspect_h5.py \
  /media/jushen/neil-liu/dataNmodels/h5_data/sub_dvt217_stack_cube_2026_0317/success_episodes

# 2. 根据输出编辑 config
#    vim src/xhum/convert/configs/dvt217_stack_cube.json

# 3. 转换（仓库根目录）
./scripts/xhum-run xhum.convert.hdf5_to_lerobot \
  --config src/xhum/convert/configs/dvt217_stack_cube.json \
  --repo_id dvt217_stack_cube \
  --src_root /media/jushen/neil-liu/dataNmodels/h5_data/sub_dvt217_stack_cube_2026_0317/success_episodes \
  --tgt_path /media/jushen/neil-liu/dataNmodels/lerobot_dataset/dvt217/dvt217_stack_cube_2026_0317 \
  --task_name stack_cube

# 4. (可选) 如果 config 里配了 stats_override，加 --stats-override 启用
./scripts/xhum-run xhum.convert.hdf5_to_lerobot \
  --config ... \
  --stats-override
```

---

## 辅助脚本与示例 Config

**版本库内**：

| 文件 | 说明 |
|------|------|
| `convert.example.sh` | 单任务转换模板 |
| `configs/dvt217_stack_cube.json` / `dvt228_grasp_water.json` | 常用 robot config 示例 |
| `configs/tianshu_72_express_demo_pi05_puppet.json` | π₀.₅ 三相机 + `divide_by` 处理夹爪缩放100倍 |
