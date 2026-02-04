#!/bin/bash
set -euo pipefail

# -------------------- activate virtual environment --------------------
if [ -d ".venv/bin" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Warning: Virtual environment not found at .venv/bin"
fi

ENGINE=${1:-vllm}
#====================================================================
# basic config
NNODES=1
# check the number of gpus
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA driver or cuda not installed"
    exit 1
fi
N_GPUS=$(nvidia-smi --list-gpus | wc -l)

REWARD_FILE="$(pwd)/utils/reward.py"
FUNCTION_NAME="compute_score"

export WANDB_PROJECT="BiPS"
EXPERIMENT_NAME="BiPS_Qwen2.5VL_Stage1"
export WANDB_RUN_ID="$EXPERIMENT_NAME-$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_NAME="$EXPERIMENT_NAME"

export WANDB_API_KEY=${WANDB_API_KEY:-"None"}
export API_KEY=${API_KEY:-"None"}

export MIN_PIXELS=$((128*128))
export MAX_PIXELS=$((2048*2048))

train_files="/mnt/data/BiPS/bips_stage1_train.parquet"
val_files="/mnt/data/BiPS/val.parquet"

use_kl_cons=true
kl_cons_coef=0.01
kl_cons_clip=1.0
image_pres_key=images_pres

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_cons=$use_kl_cons \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.image_pres_key=$image_pres_key \
    data.return_multi_modal_inputs=True \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_cons=$use_kl_cons \
    actor_rollout_ref.actor.kl_cons_coef=$kl_cons_coef \
    actor_rollout_ref.actor.kl_cons_clip=$kl_cons_clip \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.temperature=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.val_before_train=True \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$FUNCTION_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$NNODES \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.log_val_generations=3 \
    trainer.total_epochs=5 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME $@

