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

# Reproduces tab:attn_sink_mass (paper Table 1):
#   Attention mass the last-prompt-token places on the Attention Sink token set
#   (first token + current token), averaged over all layers, heads, and prompts.
#
# The compute script prints, per model:
#   "Overall average sink-set attention mass: X"  -> the "1st + current (%)" column
#
# Self-contained: each process loads its own model + dataset and runs one hooked
# forward pass. It does NOT depend on any output from table2.sh / table3.sh.
# The 6 models are sharded round-robin over the visible GPUs (one queue per GPU).

mkdir -p logs

# ---- Run-environment preamble ----------------------------------------------
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
# Pin BLAS/OMP threads so NGPU workers don't oversubscribe a shared node.
PER=$(( $(nproc) / NGPU )); [ "$PER" -lt 1 ] && PER=1
export OMP_NUM_THREADS=$PER MKL_NUM_THREADS=$PER OPENBLAS_NUM_THREADS=$PER \
       NUMEXPR_NUM_THREADS=$PER VECLIB_MAXIMUM_THREADS=$PER
export HF_DATASETS_CACHE="./hf_ds_cache"
# Build the HF dataset cache once, up front (parallel cold-cache builds race and hang).
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-release.json', split='train')"

DATA=data/prompts-cleaned-release.json
MODELS=(meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B Qwen/Qwen3-8B-Base google/gemma-2-9b mistralai/Mistral-7B-v0.1 allenai/OLMo-2-1124-7B)
bs() { case "$1" in google/gemma-2-9b) echo 162;; *) echo 648;; esac; }   # gemma: smaller batch to fit GPU memory

# ===== Attention sink mass (per-model), sharded across GPUs =====
for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in "${!MODELS[@]}"; do
      (( i % NGPU == g )) || continue
      M="${MODELS[$i]}"; SHORT=$(basename "$M")
      echo "[gpu $g] attn mass $M"
      CUDA_VISIBLE_DEVICES=$g \
      python "src/zzj_mi/mi_module/attn_sink_pattern_stats_base_models_gen.py" \
        --model_path "$M" --data_path "$DATA" --batch_size "$(bs "$M")" \
        --should_save false > "logs/attn_mass_${SHORT}.log" 2>&1
    done
  ) &
done
wait

# ===== Assemble the table =====
echo
echo "===== tab:attn_sink_mass (1st + current, %) ====="
printf "%-16s %16s\n" "Model" "1st + current"
for SHORT in Llama-3.1-8B Qwen2.5-7B Qwen3-8B-Base gemma-2-9b Mistral-7B-v0.1 OLMo-2-1124-7B; do
  log="logs/attn_mass_${SHORT}.log"
  mass=$(grep -oP 'Overall average sink-set attention mass:\s*\K[0-9.]+' "${log}" | tail -1)
  printf "%-16s %16s\n" "${SHORT}" \
    "$(awk "BEGIN{printf \"%.1f\", ${mass:-0}*100}")"
done
