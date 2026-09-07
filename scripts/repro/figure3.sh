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

# Reproduces fig:pca_neg_dir (paper Figure 3), fig:decode_not_acc (Figure 8)
# and fig:lda_intuition (Figure 11):
#   residual-stream hidden states at the "Y" position on the 4-template
#   multi-answer data, then PCA scatter (Fig 3, 11) and 10-fold PCA+LDA
#   "not"-decoding accuracy per component (Fig 8), Llama-3.1-8B and Mistral-7B.
#
# Stage 1 (GPU): one forward pass per template
#   -> outputs_release/<model>/mi/second_last_hidden_multi_template.pt (~700 MB each)
# Stage 2 (CPU): the two viz scripts -> figs/negation_circuit/{PCA,causal_ablation}/

mkdir -p logs
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
export HF_DATASETS_CACHE="./hf_ds_cache"
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-multi-release.json', split='train')"

DATA=data/prompts-cleaned-multi-release.json
MODELS=(meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.1)
for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"; SHORT=$(basename "$M")
  CUDA_VISIBLE_DEVICES=$((i % NGPU)) \
  python src/zzj_mi/mi_module/save_last_not_B_hidden_multi_template.py \
    --model_path "$M" --data_path "$DATA" --output_path outputs_release/ \
    --batch_size 162 --should_save true > "logs/f3_hidden_${SHORT}.log" 2>&1 &
done
wait

# Figure 3 (component 23 = layer 11 after attention) and Figure 11 (components 20-31)
python src/zzj_mi/mi_module/analysis_module/inspect_last_not_B_hidden_expanded.py \
  --model_path meta-llama/Llama-3.1-8B --output_path outputs_release --figs_dir figs
# Figure 8, both models
for M in "${MODELS[@]}"; do
  python src/zzj_mi/mi_module/analysis_module/inspect_last_not_B_hidden_separability_cross_valid_max.py \
    --model_path "$M" --output_path outputs_release --figs_dir figs
done
