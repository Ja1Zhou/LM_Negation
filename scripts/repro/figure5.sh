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

# Reproduces fig:fisher_scores_layers (paper Figure 5) and
# fig:fisher_scores_layers_mistral (Figure 9):
#   LLM-annotated evidence that mid-layer attention outputs promote "not Y"
#   (top-10 LogitLens tokens, layers 10-18) and suppress "Y" (bottom-10 tokens),
#   normalized evidence count per layer, Llama-3.1-8B and Mistral-7B-v0.1.
#
# Stage 1 (GPU, ~1 min per model): cache the top-k / least-k tokens
#   -> outputs_release/<model>/mi/save_attn_outputs_{topk,leastk}_expanded.pt
# Stage 2 (API): LLM annotation through OpenRouter (openai/gpt-oss-120b, bf16-pinned,
#   T=0). 648 calls per model per mode, ~$0.05 each run at 2026-09 prices. Needs
#   OPENROUTER_API_KEY in the environment or a .env file at the repo root.
#   The annotations behind the paper ship in this repo
#   (outputs_release/<model>/mi/annotate_attn_output{,_suppression}_expanded/); the
#   annotator resumes per sample, so with those present Stage 2 makes NO API calls.
#   Delete or move those directories to re-annotate from scratch (LLM output is
#   not bit-reproducible; expect the curves to match in shape, not exactly).
# Stage 3 (CPU): viz -> figs/negation_circuit/causal_ablation/evidence_counts_promote_suppress_<model>.pdf

mkdir -p logs
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
export HF_DATASETS_CACHE="./hf_ds_cache"
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-multi-release.json', split='train')"

DATA=data/prompts-cleaned-multi-release.json
MODELS=(meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.1)
for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"; SHORT=$(basename "$M")
  (
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) python src/zzj_mi/mi_module/save_attn_outputs_topk_expanded.py \
      --model_path "$M" --data_path "$DATA" --output_path outputs_release/ --should_save true \
      > "logs/f5_topk_${SHORT}.log" 2>&1
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) python src/zzj_mi/mi_module/save_attn_outputs_leastk_expanded.py \
      --model_path "$M" --data_path "$DATA" --output_path outputs_release/ --should_save true \
      > "logs/f5_leastk_${SHORT}.log" 2>&1
  ) &
done
wait

for M in "${MODELS[@]}"; do
  SHORT=$(basename "$M")
  for MODE in promote suppress; do
    python src/zzj_mi/annotate/annotate_attn_outputs.py --mode "$MODE" \
      --model_path "$M" --data_path "$DATA" --output_path outputs_release/ \
      2>&1 | tee "logs/f5_annotate_${MODE}_${SHORT}.log" | tail -4
  done
  python src/zzj_mi/mi_module/analysis_module/inspect_promote_suppress_expanded_release.py \
    --model_path "$M" --output_path outputs_release --figs_dir figs
done
