#!/usr/bin/env bash
# Copyright 2026 Zhejian Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reproduces tab:neg_acc_comparison (paper Table 3) and tab:best_layers (Table 9):
#   Negative accuracy (%) -- Full Model vs. Attention Sink (max over layers)
#   vs. LogitLens (max over layers), for all 6 models, and the best layers.
#
# Each compute script prints, per model:
#   "Neg vanilla accuracy: X"        -> Full Model column
#   "Maximum accuracy: X at layer L" -> Attn. Sink / LogitLens column (L is 0-indexed)
#
# gemma-2-9b uses a smaller batch size to fit GPU memory.
# The 6 models are sharded round-robin over the visible GPUs (one queue per GPU).

mkdir -p logs

# ---- Run-environment preamble ----------------------------------------------
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
PER=$(( $(nproc) / NGPU )); [ "$PER" -lt 1 ] && PER=1
export OMP_NUM_THREADS=$PER MKL_NUM_THREADS=$PER OPENBLAS_NUM_THREADS=$PER \
       NUMEXPR_NUM_THREADS=$PER VECLIB_MAXIMUM_THREADS=$PER
export HF_DATASETS_CACHE="./hf_ds_cache"
# Build the HF dataset cache once, up front (parallel cold-cache builds race and hang).
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-release.json', split='train')"

DATA=data/prompts-cleaned-release.json
MODELS=(meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B Qwen/Qwen3-8B-Base google/gemma-2-9b mistralai/Mistral-7B-v0.1 allenai/OLMo-2-1124-7B)
bs() { case "$1" in google/gemma-2-9b) echo 162;; *) echo 648;; esac; }

# run_sweep <script> <log prefix>: one queue per GPU, models round-robin
run_sweep() {
  for g in $(seq 0 $((NGPU - 1))); do
    (
      for i in "${!MODELS[@]}"; do
        (( i % NGPU == g )) || continue
        M="${MODELS[$i]}"; SHORT=$(basename "$M")
        echo "[gpu $g] $2 $M"
        CUDA_VISIBLE_DEVICES=$g \
        python "$1" --model_path "$M" --data_path "$DATA" --batch_size "$(bs "$M")" \
          --should_save false > "logs/$2_${SHORT}.log" 2>&1
      done
    ) &
  done
  wait
}

# ===== Phase 1: Attention Sink sweep =====
run_sweep src/zzj_mi/mi_module/attn_sink_base_models_gen.py attn_sink
# ===== Phase 2: LogitLens sweep =====
run_sweep src/zzj_mi/mi_module/logitlens_base_gen.py logitlens

# ===== Assemble the table =====
echo
echo "===== tab:neg_acc_comparison (Negative accuracy, %) ====="
printf "%-16s %12s %12s %12s\n" "Model" "Full Model" "Attn. Sink" "LogitLens"
for SHORT in Llama-3.1-8B Qwen2.5-7B Qwen3-8B-Base gemma-2-9b Mistral-7B-v0.1 OLMo-2-1124-7B; do
  as_log="logs/attn_sink_${SHORT}.log"
  ll_log="logs/logitlens_${SHORT}.log"

  full=$(grep -oP 'Neg vanilla accuracy:\s*\K[0-9.]+' "${as_log}" | tail -1)
  sink=$(grep -oP 'Maximum accuracy:\s*\K[0-9.]+'     "${as_log}" | tail -1)
  lens=$(grep -oP 'Maximum accuracy:\s*\K[0-9.]+'     "${ll_log}" | tail -1)

  printf "%-16s %12s %12s %12s\n" "${SHORT}" \
    "$(awk "BEGIN{printf \"%.1f\", ${full:-0}*100}")" \
    "$(awk "BEGIN{printf \"%.1f\", ${sink:-0}*100}")" \
    "$(awk "BEGIN{printf \"%.1f\", ${lens:-0}*100}")"
done

# ===== Best layers (paper Table 9, tab:best_layers) =====
# The scripts print 0-indexed layers; the paper table is 1-indexed.
echo
echo "===== tab:best_layers (best layer, 1-indexed) ====="
printf "%-16s %14s %14s\n" "Model" "Attn. Sink" "LogitLens"
for SHORT in Llama-3.1-8B Qwen2.5-7B Qwen3-8B-Base gemma-2-9b Mistral-7B-v0.1 OLMo-2-1124-7B; do
  sl=$(grep -oP 'Maximum accuracy:.*at layer \K[0-9]+' "logs/attn_sink_${SHORT}.log" | tail -1)
  ll=$(grep -oP 'Maximum accuracy:.*at layer \K[0-9]+' "logs/logitlens_${SHORT}.log" | tail -1)
  printf "%-16s %14s %14s\n" "${SHORT}" "$(( ${sl:-0} + 1 ))" "$(( ${ll:-0} + 1 ))"
done
