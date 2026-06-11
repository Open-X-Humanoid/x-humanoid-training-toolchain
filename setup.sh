export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


export MODELSCOPE_CACHE=/media/users/wd/modelscope/ms_cache

export HUGGINGFACE_HUB_CACHE=/media/users/wd/hf/hf_cache
export HF_HOME=/media/users/wd/hf

export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000

# # Hugging Face Offline Mode
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1


conda activate /media/users/wd/conda_envs/pi05_lerobot

# wandb配置（可选）
export WANDB_API_KEY="cc69335a09054296d36118c6b0f63ad87d9b8d35"
export WANDB_ENTITY="714305606-peking-university"
# export WANDB_PROJECT="pi05-vlm"
# export WANDB_RUN_NAME="pi05-vlm-robopoint-test"