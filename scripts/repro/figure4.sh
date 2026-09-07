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

# Reproduces fig:attn_sink_small (paper Figure 4), fig:patch_ablate_attn_outputs
# (Figure 6) and fig:patch_ablate_attn_outputs_mistral (Figure 7):
#   windowed Attention Sink ablation and attention-output path patching swept over
#   the center layer, window sizes 1-5, on the multi-answer data (surrogate
#   negation accuracy), Llama-3.1-8B and Mistral-7B-v0.1.
#
# Stage 1 (GPU, ~30-60 min per script per model): each script internally sweeps
#   w=1..5 -> outputs_release/<model>/mi/{attn_sink_ablation,patch_attn}_multil_sweepl_w{1..5}_fix_scores_sweep.pt
# Stage 2 (CPU): viz -> figs/negation_circuit/causal_ablation/
#   attn_sink_ablation_window_truncated_<model>.pdf (Fig 4, Llama)
#   attn_out_path_patching_window_<model>.pdf + attn_sink_ablation_window_<model>.pdf (Fig 6 Llama, Fig 7 Mistral)

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
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) \
    python src/zzj_mi/mi_module/attn_sink_ablation_multil_sweepl_fix_scores_expanded.py \
      --model_path "$M" --data_path "$DATA" --output_path outputs_release/ \
      --should_save true > "logs/f4_sink_window_${SHORT}.log" 2>&1
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) \
    python src/zzj_mi/mi_module/patch_attn_multil_sweepl_fix_scores_expanded.py \
      --model_path "$M" --data_path "$DATA" --output_path outputs_release/ \
      --should_save true > "logs/f4_patch_${SHORT}.log" 2>&1
  ) &
done
wait

for M in "${MODELS[@]}"; do
  python src/zzj_mi/mi_module/analysis_module/inspect_patch_sink_multil_sweepl_expanded.py \
    --model_path "$M" --data_path "$DATA" --output_path outputs_release --figs_dir figs
done
