import dataclasses
import enum
import logging
import socket
import threading
from collections import deque
import time
from pathlib import Path

import tyro

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import prepare_observation_for_inference


import socket
import pickle
import numpy as np
import torch
import cv2

import pickle 
import struct

# robot_type = "tienkung_max"

robot_type = "tienkung_pro"  # TODO !!

# 解码后统一 resize 的目标尺寸 (H, W) # 同lerobot数据即可
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

def send_msg(sock, data: bytes):
    # 先发长度（4字节，网络字节序）
    sock.sendall(struct.pack(">I", len(data)))
    sock.sendall(data)

def recv_msg(sock):
    # 先收长度
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    # 再收数据
    return recvall(sock, msg_len)

def recvall(sock, n):
    """保证收满 n 个字节"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


import threading
import time
import numpy as np
from collections import deque


def predict_action_chunk_numpy(
    obs: dict,
    task: str | None,
    model_action_steps: int = 50,
    subsample_stride: int = 2,
) -> np.ndarray:
    """
    调用 predict_action_chunk 获取完整动作序列，由外部自行维护 action_queue。
    不使用 select_action，避免写入 policy 内部 _action_queue。

    Args:
        model_action_steps: 模型原始输出步数（Pi05 默认 50）
        subsample_stride: 时间维下采样步长，2 表示 [::2] 加速

    Returns:
        np.ndarray, shape (model_action_steps // subsample_stride, action_dim)
    """
    import copy

    obs_for_inference = copy.deepcopy(obs)
    obs_for_inference = prepare_observation_for_inference(obs_for_inference, device, task=task)
    obs_for_inference = preprocessor(obs_for_inference)

    action_chunk = policy.predict_action_chunk(obs_for_inference)  # (B, chunk_size, action_dim)
    if action_chunk.ndim == 2:
        action_chunk = action_chunk.unsqueeze(0)
    action_chunk = action_chunk[:, :model_action_steps, :]

    processed_actions = []
    for i in range(action_chunk.shape[1]):
        processed_actions.append(postprocessor(action_chunk[:, i, :]))
    actions = torch.stack(processed_actions, dim=1).squeeze(0).cpu().numpy()  # (50, 19)

    if subsample_stride > 1:
        actions = actions[::subsample_stride]  # (25, 19)

    return actions


class VLASHAsyncPi0Predictor:
    """
    适配「绝对目标动作」的VLASH异步预测器
    模型输出：绝对目标动作（如目标关节角度、目标末端位姿）
    VLASH核心：推演动作执行后的未来目标状态，让模型针对该状态生成新动作
    """
    def __init__(
        self,
        policy,
        model_action_steps=50,
        subsample_stride=2, # TODO ！
        inference_trigger_threshold=15,
    ):
        """
        Args:
            policy: Pi0策略模型（输出绝对目标动作，观测需包含机器人当前状态）
            model_action_steps: 模型原始输出步数（Pi05 默认 50）
            subsample_stride: 时间维下采样步长，2 表示 [::2] → 有效 25 步
            inference_trigger_threshold: 当队列剩余N个action时启动推理
        """
        self.policy = policy
        self.model_action_steps = model_action_steps
        self.subsample_stride = subsample_stride
        self.action_horizon = model_action_steps // subsample_stride if subsample_stride > 1 else model_action_steps
        self.inference_trigger_threshold = inference_trigger_threshold
        
        # 基础队列与线程控制
        self.current_action_queue = deque()
        self.inference_thread = None
        self.inference_running = False
        self.lock = threading.Lock()
        self.consume_action_num = 0
        self.total_consume_action_num = 0
        self.action_update = False
        
        # ===== VLASH 适配绝对目标动作的核心变量 =====
        self.inference_start_state = None  # 推理启动时的机器人当前状态
        self.executed_target_states = deque()  # 已执行的目标动作（即已消耗的未来状态）

    def _predict_future_execution_state(self):
        """
        适配绝对目标动作的未来状态推演
        逻辑：
        1. 推理延迟期间，机器人会执行self.consume_action_num个目标动作（绝对量）
        2. 这些目标动作本身就是「执行后的状态」，直接取最后一个已消耗目标动作作为未来状态
        3. 动作格式: [left_arm(7), left_hand(1), right_arm(7), right_hand(1), head(3)] = 19维
           与 observation.state 格式一致，可直接使用
        """
        if self.consume_action_num == 0 or len(self.executed_target_states) == 0:
            # 无已消耗动作 → 返回推理启动时的初始状态
            return self.inference_start_state
        
        # 取推理延迟期间最后一个已执行的目标动作（即动作块执行时的机器人状态）
        # 动作本身就是绝对目标状态，格式与 observation.state 一致
        return self.executed_target_states[-1]

    def _single_inference_worker(self, obs):
        """
        改造后的推理线程：适配绝对目标动作的VLASH逻辑
        """
        try:
            # 提取 task（prompt）从 obs，避免 prepare_observation_for_inference 处理字符串
            task = obs.pop("task", None)

            # ===== VLASH Step 1: 记录推理启动状态 + 推演未来执行状态 =====
            self.inference_start_state = obs.get("observation.state", None)
            future_exec_state = self._predict_future_execution_state()
            
            # ===== VLASH Step 2: 替换观测状态为未来执行状态 =====
            vlash_obs = obs.copy()
            if future_exec_state is not None:
                vlash_obs["observation.state"] = future_exec_state
            
            # ===== 执行推理（输出绝对目标动作 chunk）=====
            new_target_actions = predict_action_chunk_numpy(
                vlash_obs, task, self.model_action_steps, self.subsample_stride
            )
            # 模型 (50, 19) → [::2] → (25, 19)
            # ===== 保留原动作平滑逻辑（适配绝对目标动作）=====
            with self.lock:
                new_remaining_actions = list(new_target_actions[self.consume_action_num:])
                old_remaining_actions = list(self.current_action_queue)
                
                # 绝对目标动作的平滑：加权平均（避免目标突变）
                if self.consume_action_num > 0 and len(old_remaining_actions) > 0:
                    exp_weight = 1
                    num_steps = min(len(old_remaining_actions), len(new_remaining_actions))
                    if num_steps > 1:
                        weights = np.exp(-exp_weight * np.arange(num_steps) / (num_steps - 1))
                        smooth_remaining_actions = [
                            (old * weights[i] + new * (1 - weights[i]))  # 绝对动作加权平均
                            for i, (old, new) in enumerate(zip(old_remaining_actions[:num_steps], new_remaining_actions[:num_steps]))
                        ] + new_remaining_actions[num_steps:]
                    else:
                        smooth_remaining_actions = new_remaining_actions
                else:
                    smooth_remaining_actions = new_remaining_actions
                
                self.current_action_queue = deque(smooth_remaining_actions)
                self.executed_target_states.clear()  # 重置已执行目标状态缓存
                print(f"[VLASH-Async] Inference done (绝对目标动作), future state: {future_exec_state}, queue size: {len(self.current_action_queue)}")
                self.consume_action_num = 0
                self.total_consume_action_num = 0
                self.action_update = True
                
        except Exception as e:
            import traceback
            print(f"[VLASH-Async] Inference FAILED: {e}")
            print(f"[VLASH-Async] Traceback:\n{''.join(traceback.format_exc())}")
        finally:
            self.inference_running = False

    def get_next_action(self, current_obs):
        """
        获取下一个绝对目标动作（新增：记录已执行的目标状态）
        """
        # 触发推理逻辑不变
        if not self.inference_running and len(self.current_action_queue) <= self.inference_trigger_threshold:
            print(f"[VLASH-Async] Triggering inference, queue size: {len(self.current_action_queue)}")
            self.inference_running = True
            self.inference_thread = threading.Thread(
                target=self._single_inference_worker,
                args=(current_obs,)
            )
            self.inference_thread.daemon = True
            self.inference_thread.start()
        
        # 等待动作队列（带推理失败恢复机制）
        wait_count = 0
        max_wait_per_attempt = 300  # 每次最多等 300 * 0.05s = 15s
        while len(self.current_action_queue) == 0:
            # 检查推理线程是否已结束且队列仍为空 → 推理失败，需要重新触发
            if not self.inference_running and len(self.current_action_queue) == 0:
                print(f"[VLASH-Async] Inference thread finished but queue is empty (inference likely failed), will retry...")
                # 不再 raise，而是返回 None 让调用方决定如何处理
                return None

            print(f"[VLASH-Async] Waiting for inference result... ({wait_count})")
            time.sleep(0.05)
            wait_count += 1
            if wait_count > max_wait_per_attempt:
                print(f"[VLASH-Async] Inference timeout after {wait_count * 0.05:.1f}s, returning None")
                return None
        
        # 获取并记录已执行的目标动作（用于未来状态推演）
        with self.lock:
            if len(self.current_action_queue) > 0:
                target_action = self.current_action_queue.popleft()
                # 记录已执行的目标动作（即机器人即将到达的状态）
                # 动作本身就是绝对目标状态，直接记录
                self.executed_target_states.append(target_action)
                
                if self.inference_running:
                    self.consume_action_num += 1
                else:
                    self.consume_action_num = 0
                self.total_consume_action_num += 1
                
                print(f"[VLASH-Async] Get target action, queue remaining: {len(self.current_action_queue)}, consumed in inference: {self.consume_action_num}")
                return target_action
        
        return None

    def reset(self):
        """重置（含已执行目标状态缓存）"""
        with self.lock:
            self.current_action_queue.clear()
            self.executed_target_states.clear()
            self.consume_action_num = 0
            self.total_consume_action_num = 0
            self.action_update = False
            self.inference_start_state = None
        
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        self.inference_running = False
        policy.reset()
        print("[VLASH-Async] Predictor reset (绝对目标动作模式)")


class AsyncPi0Predictor:
    """异步动作预测器，用于Pi0模型"""
    def __init__(
        self,
        policy,
        model_action_steps=50,
        subsample_stride=2,
        inference_trigger_threshold=15,
    ):
        """
        Args:
            policy: Pi0策略模型
            model_action_steps: 模型原始输出步数（Pi05 默认 50）
            subsample_stride: 时间维下采样步长，2 表示 [::2] → 有效 25 步
            inference_trigger_threshold: 当队列剩余N个action时启动推理
        """
        self.policy = policy
        self.model_action_steps = model_action_steps
        self.subsample_stride = subsample_stride
        self.action_horizon = model_action_steps // subsample_stride if subsample_stride > 1 else model_action_steps
        self.inference_trigger_threshold = inference_trigger_threshold
        
        self.current_action_queue = deque()
        self.inference_thread = None
        self.inference_running = False
        self.lock = threading.Lock()
        self.consume_action_num = 0
        self.total_consume_action_num = 0
        self.action_update = False
        
    def _single_inference_worker(self, obs):
        """推理工作线程"""
        try:
            # 提取 task（prompt）从 obs
            task = obs.pop("task", None)
            new_actions = predict_action_chunk_numpy(
                obs, task, self.model_action_steps, self.subsample_stride
            )
            # 模型 (50, 19) → [::2] → (25, 19)
            # 更新队列
            with self.lock:
                # 需要跳过已经消耗的action
                new_remaining_actions = list(new_actions[self.consume_action_num:])
                old_remaining_actions = list(self.current_action_queue)
                
                # 指数加权平滑
                if self.consume_action_num > 0 and len(old_remaining_actions) > 0:
                    delay_action_num = self.consume_action_num
                    total_execute_action_num = self.total_consume_action_num
                    
                    exp_weight = 1  # 调整这个值控制衰减速度
                    num_steps = min(len(old_remaining_actions), len(new_remaining_actions))
                    if num_steps > 1:
                        weights = np.exp(-exp_weight * np.arange(num_steps) / (num_steps - 1))
                        smooth_remaining_actions = [
                            (old * weights[i] + new * (1 - weights[i]))
                            for i, (old, new) in enumerate(zip(old_remaining_actions[:num_steps], new_remaining_actions[:num_steps]))
                        ] + new_remaining_actions[num_steps:]
                    else:
                        smooth_remaining_actions = new_remaining_actions
                else:
                    smooth_remaining_actions = new_remaining_actions
                
                self.current_action_queue = deque(smooth_remaining_actions)
                print(f"[Async] Inference done, consume_action_num: {self.consume_action_num}, queue updated with {len(self.current_action_queue)} actions")
                self.consume_action_num = 0
                self.total_consume_action_num = 0
                self.action_update = True
                
        except Exception as e:
            print(f"[Async] Inference error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.inference_running = False
    
    def get_next_action(self, current_obs):
        """
        获取下一个动作
        Args:
            current_obs: 当前观测（用于触发新推理）
        Returns:
            action: numpy数组，单个动作
        """
        # 当队列剩余action数量小于等于阈值时启动推理
        if not self.inference_running and len(self.current_action_queue) <= self.inference_trigger_threshold:
            print(f"[Async] Triggering inference, queue size: {len(self.current_action_queue)}")
            self.inference_running = True
            self.inference_thread = threading.Thread(
                target=self._single_inference_worker,
                args=(current_obs,)
            )
            self.inference_thread.daemon = True
            self.inference_thread.start()
        
        # 等待至少有一个动作（带推理失败恢复机制）
        wait_count = 0
        max_wait_per_attempt = 300  # 每次最多等 300 * 0.05s = 15s
        while len(self.current_action_queue) == 0:
            # 检查推理线程是否已结束且队列仍为空 → 推理失败，需要重新触发
            if not self.inference_running and len(self.current_action_queue) == 0:
                print(f"[Async] Inference thread finished but queue is empty (inference likely failed), will retry...")
                return None

            print(f"[Async] Waiting for inference result... ({wait_count})")
            time.sleep(0.05)
            wait_count += 1
            if wait_count > max_wait_per_attempt:
                print(f"[Async] Inference timeout after {wait_count * 0.05:.1f}s, returning None")
                return None
        
        # 从队列中获取下一个动作
        with self.lock:
            if len(self.current_action_queue) > 0:
                action = self.current_action_queue.popleft()
                # 只有在推理线程运行时才计算consume_action_num
                if self.inference_running:
                    self.consume_action_num += 1
                else:
                    self.consume_action_num = 0
                self.total_consume_action_num += 1
                print(f"[Async] Get action, queue remaining: {len(self.current_action_queue)}, inference_running: {self.inference_running}")
                return action
        
        return None
    
    def reset(self):
        """重置预测器状态"""
        with self.lock:
            self.current_action_queue.clear()
            self.consume_action_num = 0
            self.total_consume_action_num = 0
            self.action_update = False
        # 等待推理线程结束
        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)
        self.inference_running = False
        policy.reset()
        print("[Async] Predictor reset")


# 加载模型 - 使用 from_pretrained 方式
# 注意：config.json 和 model.safetensors 在 pretrained_model 子目录下
# TODO !
# multi_train_pi05_tianshu_72_all_white_black_upward_downward_unfreeze_vision_encoder_0607
pretrained_path = Path("/home/eai/Dev/wd/ckpt/pi05/multi_train_pi05_tianshu_72_all_white_black_upward_downward_unfreeze_vision_encoder_0607/checkpoints/015000/015000/pretrained_model")

# 使用 policy_class.from_pretrained() 加载模型（与训练代码一致）
policy = PI05Policy.from_pretrained(pretrained_path)

# 加载处理器
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=pretrained_path,
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)


# TODO !!
# HOST = "10.11.10.122" 
HOST = "192.168.41.1"
# HOST = "10.11.186.21"

PORT = 9000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# 创建异步预测器
# model_action_steps=50 模型原始输出，subsample_stride=2 时间维 [::2] → 队列有效 25 步
# inference_trigger_threshold 建议为 action_horizon(25) 的 20%-40%，即约 10~15
async_predictor = VLASHAsyncPi0Predictor(
    policy=policy,
    model_action_steps=50,
    subsample_stride=1, # TODO ！！！
    inference_trigger_threshold=30,
)


print("===== VLASHAsync Pi0 Inference Client Started =====")

# 动作缓冲区，积累5个动作后再发送
action_buffer = []
BATCH_SIZE = 3 # 5 # TODO !!

while True:
    # 1. 接收 obs 或 reset 信号
    data = recv_msg(sock)
    if data is None:
        print("Server disconnected.")
        # 发送剩余的缓冲动作
        if len(action_buffer) > 0:
            action_list = [action.tolist() if isinstance(action, np.ndarray) else action for action in action_buffer]
            send_msg(sock, pickle.dumps(action_list))
            print(f"[Client] Sent remaining {len(action_buffer)} action(s) before disconnect")
        break
    
    received_data = pickle.loads(data)
    
    # 检查是否是reset信号
    if isinstance(received_data, dict) and received_data.get("type") == "reset":
        print("[Client] Received reset signal, resetting async predictor...")
        # 不再发送剩余缓冲动作，直接丢弃
        if len(action_buffer) > 0:
            print(f"[Client] Discarding {len(action_buffer)} buffered action(s) on reset")
            action_buffer.clear()
        async_predictor.reset()
        # 发送确认信号
        ack_signal = {"type": "reset_ack"}
        send_msg(sock, pickle.dumps(ack_signal))
        print("[Client] Reset complete, waiting for new observations...")
        continue
    
    # 正常的obs数据
    obs_data = received_data
    
    # 解压缩图像
    decoded_images = {}
    for cam_name, buf in obs_data["images"].items():
        img_array = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"[Client] Decoded image '{cam_name}': shape={img.shape} (H, W, C)")
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        decoded_images[cam_name] = img
        print(f"[Client] Resized image '{cam_name}': shape={img.shape} (H, W, C)")

    # 2. 组装输入
    # 服务器相机名 → pi05模型图像key 的映射
    if robot_type == "tienkung_max":
        camera_name_map = {
            "ob_camera_head": "observation.images.base_0_rgb",
            "ob_camera_left": "observation.images.left_wrist_0_rgb",
            "ob_camera_right": "observation.images.right_wrist_0_rgb",
        }
    elif robot_type == "tienkung_pro":
        camera_name_map = {
            "head": "observation.images.base_0_rgb",
            "left": "observation.images.left_wrist_0_rgb",
            "right": "observation.images.right_wrist_0_rgb",
        }

    # 组装 flat observation（适配 prepare_observation_for_inference）
    re_obs = {
        "task": obs_data["prompt"],  # prompt 作为 task 传入
        "observation.state": obs_data["state"],  # 19维: 左臂7+左手1+右臂7+右手1+头部3
    }

    # 添加3个相机图像（保持 HWC 格式，prepare_observation_for_inference 内部会做 permute）
    for server_cam_name, pi05_key in camera_name_map.items():
        if server_cam_name in decoded_images:
            # re_obs[pi05_key] = np.transpose(decoded_images[server_cam_name], (2, 0, 1))
            re_obs[pi05_key] = decoded_images[server_cam_name]
        else:
            print(f"[Client] Warning: camera '{server_cam_name}' not found in server data, available: {list(decoded_images.keys())}")

    print(f"[Client] Prompt: {re_obs['task']}")

    # 3. 异步获取动作（如果队列为空或接近空，会自动触发推理）
    pred_action = async_predictor.get_next_action(re_obs)
    
    if pred_action is None:
        print("[Client] Failed to get action, waiting before retry...")
        time.sleep(0.1)  # 避免紧密重试循环
        continue

    # 4. 将动作添加到缓冲区
    action_buffer.append(pred_action)
    print(f"[Client] Buffered action, buffer size: {len(action_buffer)}/{BATCH_SIZE}")

    # 5. 当缓冲区达到5个动作时，批量发送给 server
    if len(action_buffer) >= BATCH_SIZE:
        # 将numpy数组转为list发送
        action_list = [action.tolist() if isinstance(action, np.ndarray) else action for action in action_buffer]
        send_msg(sock, pickle.dumps(action_list))
        print(f"[Client] Sent batch of {len(action_buffer)} actions")
        action_buffer.clear()

sock.close()
print("===== Client Closed =====")
