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

"""
src/zzj_mi/olmo_module/viz_acc_n_sensitivity.py

Reproduces fig:olmo_train_accs (paper Figure 2):
  OLMo-2 Positive/Negative accuracy and Sensitivity versus pre-training step.

Reads the per-checkpoint results written by `acc_n_sensitivity.py`
(one `acc_n_sensitivity.pt` per revision under
`<output_path>/allenai--OLMo-2-1124-7B/<revision>/mi/`) and plots them
against training step (x-axis on a log scale).

Example:
    python src/zzj_mi/olmo_module/viz_acc_n_sensitivity.py \
        --output_path outputs_release/ \
        --model_path allenai/OLMo-2-1124-7B \
        --save_path figs/trace_emergence/accs_n_sensitivity.pdf
"""
import argparse
import re
from pathlib import Path

import torch as t
import matplotlib.pyplot as plt

REVISION_PATTERN = r"stage1-step(\d+)-tokens\d+B"


def parse_step(result: Path) -> int | None:
    match = re.search(REVISION_PATTERN, str(result))
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="outputs_release/")
    parser.add_argument("--model_path", default="allenai/OLMo-2-1124-7B")
    parser.add_argument("--save_path", default="figs/trace_emergence/accs_n_sensitivity.pdf")
    args = parser.parse_args()

    normalized_model_path = args.model_path.replace("/", "--")
    output_dir = Path(args.output_path) / normalized_model_path

    results = list(output_dir.glob("*/mi/acc_n_sensitivity.pt"))
    if not results:
        raise SystemExit(f"No acc_n_sensitivity.pt found under {output_dir}")

    data_with_steps = sorted(
        ((parse_step(r), t.load(r)) for r in results),
        key=lambda x: x[0],
    )
    steps, data = zip(*data_with_steps)
    neg_accs = [d["neg_acc"] for d in data]
    pos_accs = [d["pos_acc"] for d in data]
    sensitivity_accs = [d["sensitivity_acc"] for d in data]
    print(f"Loaded {len(steps)} checkpoints: steps {steps[0]}..{steps[-1]}")

    plt.figure(figsize=(6, 4), dpi=300)
    plt.rcParams.update({"font.size": 16})
    plt.plot(steps, pos_accs, label="Positive Acc", linewidth=3)
    plt.plot(steps, neg_accs, label="Negative Acc", linewidth=3)
    plt.plot(steps, sensitivity_accs, label="Sensitivity", linewidth=3)
    plt.xscale("log")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=14, loc="upper right", bbox_to_anchor=(0.98, 0.90))
    plt.grid(True, which="both", linestyle="-", alpha=0.3)
    plt.tight_layout()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved figure to: {save_path}")
