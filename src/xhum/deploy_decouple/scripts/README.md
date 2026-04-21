# `deploy_decouple/scripts` — 本地脚本说明 / Local helper scripts

**[中文](#中文)** · **[English](#english)**

除本文件外，`scripts/` 下其它文件由仓库根 `.gitignore` 忽略，需在本机自行放置 `.py`（评测、冒烟、关节对比等）。

All other files under `scripts/` are gitignored; keep your own `.py` helpers here (eval, smoke tests, joint compare, etc.).

工作目录均为 **`src/xhum/deploy_decouple`**（下文命令相对该路径）。

Working directory for commands below: **`src/xhum/deploy_decouple`**.

---

## 中文

### 前置

- 评测 / `PolicyAgent`：**Python ≥3.12**、`lerobot`、`torch`、**`h5py`** 等；`export PYTHONPATH=/你的路径/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH`
- 仅 HDF5 动作加载测试：**Python 3.10** 亦可，需 **`h5py`**
- **`compare_joints.py`**：需 **numpy**

### 命令一览

| 命令 | 作用 |
|------|------|
| `python scripts/test_replay_hdf5.py /path/to/trajectory.hdf5` | 只测 HDF5 **动作**加载（与 **`mode=replay_actions`** 一致）；**无 ZMQ**。 |
| `python scripts/test_policy_agent_fake.py --model_path .../pretrained_model` | 进程内 **`PolicyAgent`** + 随机观测测一帧（**无 ZMQ**，需 **≥3.12**）。 |
| `python scripts/eval_policy_from_hdf5.py --h5_path ... --model_path ...` | 逐帧 **`PolicyAgent.inference`**，打印 **`pred`**、**`gt_next`**（**`--gt_key`** 下一行）、**`diff(pred-gt_next)`**；结束按维度输出 mean/max/min。默认输入：**`observations/rgb_images/camera_camera`** + **`puppet/joint_position`**，**`obs_camera_key=camera`**；默认 GT：**`master/joint_position`**。均可 CLI 覆盖。**`--quiet`** 关闭逐步打印。 |

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

---

## English

### Prerequisites

- Eval / `PolicyAgent`: **Python ≥3.12**, `lerobot`, `torch`, **`h5py`**; `export PYTHONPATH=/path/to/x-humanoid-training-toolchain/lerobot/src:$PYTHONPATH`
- HDF5 actions-only test: **Python 3.10** ok, needs **`h5py`**
- **`compare_joints.py`**: needs **numpy**

### Command reference

| Command | Purpose |
|---------|---------|
| `python scripts/test_replay_hdf5.py /path/to/trajectory.hdf5` | HDF5 **actions** load only (same as **`mode=replay_actions`**); no ZMQ. |
| `python scripts/test_policy_agent_fake.py --model_path .../pretrained_model` | One in-process **`PolicyAgent`** step with random obs (no ZMQ, Py **≥3.12**). |
| `python scripts/eval_policy_from_hdf5.py --h5_path ... --model_path ...` | **`PolicyAgent.inference`** each step; prints **`pred`**, next-row **`gt_next`** from **`--gt_key`**, **`diff(pred-gt_next)`**, then per-dimension mean/max/min. Defaults: RGB **`observations/rgb_images/camera_camera`**, state **`puppet/joint_position`**, **`obs_camera_key=camera`**, GT **`master/joint_position`**. **`--quiet`**: no per-step vectors. |

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

---

主文档（ZMQ / ROS / `joints` 落盘配置）：[`../README_zh.md`](../README_zh.md) · [`../README.md`](../README.md)
