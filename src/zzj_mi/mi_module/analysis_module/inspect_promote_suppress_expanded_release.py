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
src/zzj_mi/mi_module/analysis_module/inspect_promote_suppress_expanded_release.py

Paper Figure 5 (Llama-3.1-8B) and Figure 9 (Mistral-7B-v0.1): normalized evidence
counts per attention layer from the two LLM-annotation runs
  - "not Y" evidence in promoted tokens   -> <mi>/annotate_attn_output_expanded/summary.json
  - "Y" evidence in suppressed tokens     -> <mi>/annotate_attn_output_suppression_expanded/summary.json
where <mi> = <output_path>/<model>/mi (see src/zzj_mi/annotate/annotate_attn_outputs.py).
Counts are divided by the number of annotated samples per source.

    python src/zzj_mi/mi_module/analysis_module/inspect_promote_suppress_expanded_release.py \
        --model_path meta-llama/Llama-3.1-8B --output_path outputs_release --figs_dir figs
Writes <figs_dir>/negation_circuit/causal_ablation/evidence_counts_promote_suppress_<model>.pdf
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

START_LAYER = 8
END_LAYER = 18


def parse_args():
    p = ArgumentParser()
    p.add_argument("--model_path", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--output_path", default="outputs_release")
    p.add_argument("--output_subdir", default=None)
    p.add_argument("--figs_dir", default="figs")
    return p.parse_args()


def load_evidence_counts(outputs_dir: Path, annotation_dir_name: str):
    summary_file = outputs_dir / annotation_dir_name / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Annotation summary not found: {summary_file}")
    summary = json.loads(summary_file.read_text())
    evidence = {int(k): v for k, v in summary.get("layer_evidence_counts", {}).items()}
    total = summary.get("total_processed", summary.get("total_samples"))
    print(f"Loaded {annotation_dir_name}: {len(evidence)} layers, {total} samples")
    return evidence, total


def main():
    args = parse_args()
    outputs_dir = Path(args.output_path) / args.model_path.replace("/", "--") / "mi"
    if args.output_subdir:
        outputs_dir = outputs_dir / args.output_subdir
    not_b_evidence, not_b_total = load_evidence_counts(outputs_dir, "annotate_attn_output_expanded")
    b_evidence, b_total = load_evidence_counts(outputs_dir, "annotate_attn_output_suppression_expanded")

    layers = list(range(START_LAYER, END_LAYER + 1))
    not_b_values = [not_b_evidence.get(l, 0) / not_b_total for l in layers]
    b_values = [b_evidence.get(l, 0) / b_total for l in layers]

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.rcParams.update({"font.size": 16})
    ax.plot(layers, not_b_values, color="royalblue", linewidth=2, alpha=0.7, marker="o", markersize=6,
            label='"not Y" (Promoted)')
    ax.plot(layers, b_values, color="crimson", linewidth=2, alpha=0.7, marker="s", markersize=6,
            label='"Y" (Suppressed)')
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Evidence Count (Normalized)")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], rotation=45, ha="right", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=12)
    plt.tight_layout()

    out_path = (Path(args.figs_dir) / "negation_circuit/causal_ablation"
                / f"evidence_counts_promote_suppress_{args.model_path.replace('/', '--')}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure to: {out_path}")
    for l, nb, b in zip(layers, not_b_values, b_values):
        print(f"  layer {l:2d}: not-Y {nb:.3f}  Y {b:.3f}")


if __name__ == "__main__":
    main()
