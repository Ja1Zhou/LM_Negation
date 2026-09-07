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

# Reproduces fig:olmo_train_accs (paper Figure 2):
#   OLMo-2 Positive/Negative accuracy and Sensitivity across pre-training steps.
#
# Two stages:
#   1) Sweep OLMo-2-1124-7B pre-training checkpoints (the ~28 stage1 revisions
#      listed in src/zzj_mi/olmo_module/checkpoints.txt). For each, evaluate
#      pos/neg accuracy + sensitivity on the dataset and save acc_n_sensitivity.pt
#      under outputs_release/allenai--OLMo-2-1124-7B/<revision>/mi/.
#   2) Plot all checkpoints vs. training step -> figs/trace_emergence/accs_n_sensitivity.pdf
#
# Self-contained: depends only on the dataset and on the OLMo-2 checkpoint
# branches (expected already downloaded into the HF cache). Does NOT depend on
# any output from the table scripts. Only the baseline (no attention-sink) curve
# appears in the paper figure, so only that stage is run here.
#
# Parallel: the 28 checkpoints are sharded round-robin across all visible GPUs;
# each GPU works through its own queue sequentially. Re-runnable -- any checkpoint
# whose acc_n_sensitivity.pt already exists is skipped.

mkdir -p logs

MODEL="allenai/OLMo-2-1124-7B"
NORM_MODEL="allenai--OLMo-2-1124-7B"
OUT_ROOT="outputs_release/"

# ---- Run-environment preamble ----------------------------------------------
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -lt 1 ] && NGPU=1
# Pin BLAS/OMP threads so NGPU workers don't oversubscribe a shared node.
PER=$(( $(nproc) / NGPU )); [ "$PER" -lt 1 ] && PER=1
export OMP_NUM_THREADS=$PER MKL_NUM_THREADS=$PER OPENBLAS_NUM_THREADS=$PER \
       NUMEXPR_NUM_THREADS=$PER VECLIB_MAXIMUM_THREADS=$PER
export HF_DATASETS_CACHE="./hf_ds_cache"
export HF_XET_HIGH_PERFORMANCE=1
# Build the HF dataset cache once, up front (avoid a cold-cache build race).
python -c "from datasets import load_dataset; \
load_dataset('json', data_files='data/prompts-cleaned-release.json', split='train')"

# ---- Resolve the stage1 branch names for the target steps ------------------
mapfile -t REVS < <(python src/zzj_mi/olmo_module/generate_download_commands_from_checkpoints.py \
                    2>/dev/null | grep -oP 'revision \K\S+')
echo "Resolved ${#REVS[@]} checkpoint revisions; sharding across ${NGPU} GPU(s)."

# ===== Stage 1: checkpoint sweep (baseline), sharded across GPUs =====
for g in $(seq 0 $((NGPU - 1))); do
  (
    for idx in "${!REVS[@]}"; do
      (( idx % NGPU == g )) || continue
      rev="${REVS[idx]}"
      out="${OUT_ROOT}${NORM_MODEL}/${rev}/mi/acc_n_sensitivity.pt"
      if [ -f "$out" ]; then
        echo "[gpu $g] skip ${rev} (exists)"
        continue
      fi
      echo "[gpu $g] run ${rev}"
      CUDA_VISIBLE_DEVICES=$g \
      python "src/zzj_mi/olmo_module/acc_n_sensitivity.py" \
        --model_path "$MODEL" \
        --model_revision "$rev" \
        --data_path data/prompts-cleaned-release.json \
        --output_path "$OUT_ROOT" \
        --batch_size 648 \
        --should_save true \
        > "logs/olmo_${rev}.log" 2>&1
    done
  ) &
done

wait

# ===== Stage 2: plot the figure =====
python "src/zzj_mi/olmo_module/viz_acc_n_sensitivity.py" \
  --output_path "$OUT_ROOT" \
  --model_path "$MODEL" \
  --save_path "figs/trace_emergence/accs_n_sensitivity.pdf"
