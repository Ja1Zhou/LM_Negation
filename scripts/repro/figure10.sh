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

# Reproduces fig:attn_sink_neg_pos_sweep (paper Figure 10):
#   negative and positive accuracy as a function of the layer from which
#   Cumulative Attention Sink is applied, all 6 models (single-answer data).
#
# Stage 1 (GPU): the Table 3 attention-sink sweep re-run with --should_save true
#   (negative prompts) plus its positive-prompt counterpart; the 6 models sharded
#   over the visible GPUs -> outputs_release/<model>/mi/{attn_sink_base_models_gen_neg_acc,attn_sink_pos_control_sweep_pos_acc}.pkl
# Stage 2 (CPU): viz -> figs/negation_circuit/causal_ablation/all_models_neg_pos_sweep.pdf
#   (the dotted vanilla lines are constants in the viz script, taken from the logs of Table 2/3)

mkdir -p logs
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
PER=$(( $(nproc) / NGPU )); [ "$PER" -lt 1 ] && PER=1
export OMP_NUM_THREADS=$PER MKL_NUM_THREADS=$PER OPENBLAS_NUM_THREADS=$PER \
       NUMEXPR_NUM_THREADS=$PER VECLIB_MAXIMUM_THREADS=$PER
export HF_DATASETS_CACHE="./hf_ds_cache"
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-release.json', split='train')"

DATA=data/prompts-cleaned-release.json
MODELS=(meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B Qwen/Qwen3-8B-Base google/gemma-2-9b mistralai/Mistral-7B-v0.1 allenai/OLMo-2-1124-7B)
bs() { case "$1" in google/gemma-2-9b) echo 162;; *) echo 648;; esac; }   # gemma: smaller batch to fit GPU memory
for g in $(seq 0 $((NGPU - 1))); do
  (
    for i in "${!MODELS[@]}"; do
      (( i % NGPU == g )) || continue
      M="${MODELS[$i]}"; SHORT=$(basename "$M")
      echo "[gpu $g] sweeps $M"
      CUDA_VISIBLE_DEVICES=$g python src/zzj_mi/mi_module/attn_sink_base_models_gen.py \
        --model_path "$M" --data_path "$DATA" --batch_size "$(bs "$M")" \
        --output_path outputs_release/ --should_save true > "logs/f10_neg_sweep_${SHORT}.log" 2>&1
      CUDA_VISIBLE_DEVICES=$g python src/zzj_mi/mi_module/attn_sink_pos_control_sweep_base_models_gen.py \
        --model_path "$M" --data_path "$DATA" --batch_size "$(bs "$M")" \
        --output_path outputs_release/ --should_save true > "logs/f10_pos_sweep_${SHORT}.log" 2>&1
    done
  ) &
done
wait

python src/zzj_mi/mi_module/analysis_module/viz_attn_sink_neg_pos_per_layer.py \
  --output_path outputs_release --figs_dir figs
