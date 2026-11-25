#!/usr/bin/env bash
# Reference for physical 950DT 96GB x 8 and matching GLM-5.2 W4A4C8/MXFP4.
# The CPU benchmark uses bench.py, independently of this launcher.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to your GLM-5.2-w4a4c8-mxfp4 checkpoint}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=400
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
exec vllm serve "$MODEL_PATH" \
  --host "${BIND_HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --api-server-count 1 --data-parallel-size 1 --tensor-parallel-size 8 \
  --enable-expert-parallel --max-model-len 135000 \
  --max-num-batched-tokens 16384 --max-num-seqs 20 \
  --served-model-name glm --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --trust-remote-code --enforce-eager --quantization ascend \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype fp8 \
  --additional-config '{"enable_flashcomm1":true,"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"enable_dsa_cp":true}'
