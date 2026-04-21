# `deploy_decouple/scripts` — 本地脚本说明 / Local helper scripts

**[中文](#中文)** · **[English](#english)**

**已纳入版本库**：`README.md`、`stat_hdf5_firstframe_mean.py`。其余 `*.py` 由仓库根 `.gitignore` 忽略，需在本机自行放置（如评测、关节对比等）。

**Tracked in git:** `README.md`, `stat_hdf5_firstframe_mean.py`. Other `*.py` here are gitignored; keep local copies (eval, joint compare, etc.).

工作目录均为 **`src/xhum/deploy_decouple`**（下文命令相对该路径）。

Working directory for commands below: **`src/xhum/deploy_decouple`**.

---

## 中文

### 前置

- 评测 / `PolicyAgent`：**Python ≥3.12**、`lerobot`、`torch`、**`h5py`** 等；`export PYTHONPATH=/你的路径/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH`
- **`compare_joints.py`**：需 **numpy**
- **`stat_hdf5_firstframe_mean.py`**：**Python ≥3.10**，**`numpy`**、**`h5py`**（无需 LeRobot）

### 命令一览

| 命令 | 作用 |
|------|------|
| `python scripts/eval_policy_from_hdf5.py --h5_path ... --model_path ...` | 逐帧 **`PolicyAgent.inference`**，打印 **`pred`**、**`gt_next`**（**`--gt_key`** 下一行）、**`diff(pred-gt_next)`**；结束按维度输出 mean/max/min。默认输入：**`observations/rgb_images/camera_camera`** + **`puppet/joint_position`**，**`obs_camera_key=camera`**；默认 GT：**`master/joint_position`**。均可 CLI 覆盖。**`--quiet`** 关闭逐步打印。 |
| `python scripts/stat_hdf5_firstframe_mean.py --dir ... --key puppet/joint_position` | 遍历目录下所有 **`.hdf5`/`.h5`**，对每个文件取 **`--key`** 的 **第 0 帧**（`dataset[0]`），在文件维上做逐元素 **平均**；写入 **`--output`** 的 **`.npy`**（均值数组）与 **`.json`**（路径列表、跳过原因等）。可加 **`-r`** 递归子目录；**`--strict`** 任一文件失败则退出非 0。 |

### HDF5 离线推理示例

```bash
export PYTHONPATH=/你的路径/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH
python scripts/eval_policy_from_hdf5.py \
  --h5_path /path/to/trajectory.hdf5 \
  --model_path /path/to/pretrained_model \
  --max_steps 100
```

可选 **`--gt_key`**、**`--replay_images_h5_key`**、**`--replay_state_h5_key`**、**`--obs_camera_key`**（须与 checkpoint 视觉短名一致）、**`--start`**。**`--max_steps 0`** 跑满。**`--quiet`** 不打印每步向量。

### 关节落盘后的对比（`compare_joints.py`）

与主 README「关节向量落盘与校准」一节配合：在客户端 **`client_pre_send`** 与服务端 **`server_post_decode`** 各有一批 `state_XXXXXXXX.npy` 后，运行：

```bash
python3 scripts/compare_joints.py \
  --client_dir /path/to/.../client_pre_send \
  --server_dir /path/to/.../server_post_decode
```

可选：**`--atol`**、**`--rtol`**（默认约 `1e-5`）、**`--verbose`**。终端打印 **`PASS`/`FAIL`**；**退出码 0** 表示序号对齐且 **`np.allclose`** 通过。

### 多文件 HDF5 首帧均值（`stat_hdf5_firstframe_mean.py`）

```bash
cd src/xhum/deploy_decouple
python3 scripts/stat_hdf5_firstframe_mean.py \
  --dir /path/to/folder_with_many_h5 \
  --key puppet/joint_position \
  --output /path/to/out/puppet_first_mean
# 生成 puppet_first_mean.npy 与 puppet_first_mean.json；未指定 --output 时写到当前目录 firstframe_mean_<key>.npy
```

---

## English

### Prerequisites

- Eval / `PolicyAgent`: **Python ≥3.12**, `lerobot`, `torch`, **`h5py`**; `export PYTHONPATH=/path/to/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH`
- **`compare_joints.py`**: needs **numpy**
- **`stat_hdf5_firstframe_mean.py`**: **Python ≥3.10**, **`numpy`**, **`h5py`** (no LeRobot)

### Command reference

| Command | Purpose |
|---------|---------|
| `python scripts/eval_policy_from_hdf5.py --h5_path ... --model_path ...` | **`PolicyAgent.inference`** each step; prints **`pred`**, next-row **`gt_next`** from **`--gt_key`**, **`diff(pred-gt_next)`**, then per-dimension mean/max/min. Defaults: RGB **`observations/rgb_images/camera_camera`**, state **`puppet/joint_position`**, **`obs_camera_key=camera`**, GT **`master/joint_position`**. **`--quiet`**: no per-step vectors. |
| `python scripts/stat_hdf5_firstframe_mean.py --dir ... --key puppet/joint_position` | For every **`.hdf5`/`.h5`** under **`--dir`**, read **`dataset[0]`** at **`--key`**, element-wise **mean** over files; write **`--output`**.**`npy`** and **`.json`** (paths used, skips). **`-r`**: recursive; **`--strict`**: fail on any skip. |

### HDF5 offline eval example

```bash
export PYTHONPATH=/path/to/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH
python scripts/eval_policy_from_hdf5.py \
  --h5_path /path/to/trajectory.hdf5 \
  --model_path /path/to/pretrained_model \
  --max_steps 100
```

Optional **`--gt_key`**, **`--replay_images_h5_key`**, **`--replay_state_h5_key`**, **`--obs_camera_key`**, **`--start`**. **`--max_steps 0`** runs all frames. **`--quiet`** skips per-step prints.

### Joint dump compare (`compare_joints.py`)

After enabling joint dumps in YAML / `policy_server` (see main README **Joint dumps & calibration**), compare **`client_pre_send`** vs **`server_post_decode`** `state_*.npy` trees:

```bash
python3 scripts/compare_joints.py \
  --client_dir /path/to/.../client_pre_send \
  --server_dir /path/to/.../server_post_decode
```

Optional **`--atol`**, **`--rtol`** (~`1e-5` default), **`--verbose`**. Exit code **0** = all aligned steps pass **`np.allclose`**.

### First-frame mean over many HDF5 (`stat_hdf5_firstframe_mean.py`)

```bash
cd src/xhum/deploy_decouple
python3 scripts/stat_hdf5_firstframe_mean.py \
  --dir /path/to/folder_with_many_h5 \
  --key puppet/joint_position \
  --output /path/to/out/puppet_first_mean
# writes puppet_first_mean.npy + puppet_first_mean.json; default --output stem: ./firstframe_mean_<key>
```

---

主文档（ZMQ / ROS / `joints` 落盘配置）：[`../README_zh.md`](../README_zh.md) · [`../README.md`](../README.md)
