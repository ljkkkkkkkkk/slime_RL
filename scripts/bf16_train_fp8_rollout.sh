#!/bin/bash
export WANDB_API_KEY="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"
export WANDB_KEY=$WANDB_API_KEY


pkill -9 sglang || true
sleep 2
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 2
pkill -9 ray || true
pkill -9 python || true

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT:-0}" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="/lianjiakun1"
SLIME_ROOT="/lianjiakun1/slime"

source "/lianjiakun1/slime/scripts/models/qwen2.5-7B.sh"

HF_FP8="/lianjiakun1/models/Qwen2.5-Math-7B-FP8"
TORCH_DIST="/lianjiakun1/models/Qwen2.5-Math-7B_torch_dist"
EXP_DIR="/lianjiakun1/exp/qwen25math7b_int8"
LOG_DIR="/lianjiakun1/logs"
# 检查点参数
CKPT_ARGS=(
  --hf-checkpoint "${HF_FP8}"
  --ref-load "${TORCH_DIST}"
  --load "${EXP_DIR}"
  --save "${EXP_DIR}"
  --save-interval 20
)

# 生成参数
ROLLOUT_ARGS=(
  --prompt-data /lianjiakun1/data/math/math/train.jsonl
  --input-key prompt
  --label-key reward_model.ground_truth
  --apply-chat-template
  --rollout-shuffle
  --custom-rm-path slime.rollout.rm_hub.reward_fn.compute_score
  --num-rollout 200
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len 2048
  --rollout-temperature 1
  --num-steps-per-rollout 2
  --balance-data
)

# 评估参数
EVAL_ARGS=(
  --eval-interval 20
  --eval-prompt-data math /lianjiakun1/data/math/math/train.jsonl
  --n-samples-per-eval-prompt 8
  --eval-max-response-len 2048
  --eval-top-p 1
  --eval-temperature 0.6
)

# 性能参数
PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 16384
  
  --bf16
)

# GRPO 参数
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
)

# 优化器参数
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98

)


# WandB 参数
WANDB_ARGS=(
  --use-wandb
  --wandb-project slime
  --wandb-group qwen25math7b-fp8
  --wandb-key "${WANDB_KEY}"
)

# SGLang 参数
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.8
)

# 杂项参数
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

mkdir -p "${EXP_DIR}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PROJECT_ROOT}:${SLIME_ROOT}:/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_KEY\": \"${WANDB_API_KEY}\"
  }
}"
export PYTHONPATH="${PROJECT_ROOT}:${SLIME_ROOT}:$PYTHONPATH"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 /lianjiakun1/slime/train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} 2>&1 | tee -a "${LOG_FILE}"
