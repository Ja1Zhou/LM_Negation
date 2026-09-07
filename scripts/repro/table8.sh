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

# Reproduces tab:multi_ans (paper Table 8, Appendix):
#   Neg Acc, Pos Acc, Sensitivity (%) with logits averaged over the multiple
#   candidate answers per prompt (data/prompts-cleaned-multi-release.json).

mkdir -p logs

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
PER=$(( $(nproc) / NGPU )); [ "$PER" -lt 1 ] && PER=1
export OMP_NUM_THREADS=$PER MKL_NUM_THREADS=$PER OPENBLAS_NUM_THREADS=$PER \
       NUMEXPR_NUM_THREADS=$PER VECLIB_MAXIMUM_THREADS=$PER
export HF_DATASETS_CACHE="./hf_ds_cache"
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-multi-release.json', split='train')"

MODELS=(meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B Qwen/Qwen3-8B-Base google/gemma-2-9b mistralai/Mistral-7B-v0.1 allenai/OLMo-2-1124-7B)
for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in "${!MODELS[@]}"; do
      (( i % NGPU == g )) || continue
      M="${MODELS[$i]}"; SHORT=$(basename "$M")
      echo "[gpu $g] table8 $M"
      CUDA_VISIBLE_DEVICES=$g \
      python "src/zzj_mi/mi_module/logits_base_models_gen_expanded.py" \
        --model_path "$M" --data_path data/prompts-cleaned-multi-release.json \
        --batch_size 324 --should_save false > "logs/table8_${SHORT}.log" 2>&1
    done
  ) &
done
wait

echo
echo "===== tab:multi_ans (multi-answer, %) ====="
printf "%-16s %10s %10s %12s\n" "Model" "Neg Acc" "Pos Acc" "Sensitivity"
for M in "${MODELS[@]}"; do
  SHORT=$(basename "$M"); log="logs/table8_${SHORT}.log"
  neg=$(grep -oP '^Neg accuracy:\s*\K[0-9.]+'         "$log" | tail -1)
  pos=$(grep -oP '^Pos accuracy:\s*\K[0-9.]+'         "$log" | tail -1)
  sen=$(grep -oP '^Sensitivity accuracy:\s*\K[0-9.]+' "$log" | tail -1)
  printf "%-16s %10s %10s %12s\n" "${SHORT}" \
    "$(awk "BEGIN{printf \"%.1f\", ${neg:-0}*100}")" \
    "$(awk "BEGIN{printf \"%.1f\", ${pos:-0}*100}")" \
    "$(awk "BEGIN{printf \"%.1f\", ${sen:-0}*100}")"
done
