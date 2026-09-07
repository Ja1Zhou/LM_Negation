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
Paper Figure 10 (fig:attn_sink_neg_pos_sweep): negative and positive accuracy
vs. the layer from which Cumulative Attention Sink is applied, all 6 models.

Inputs, all written by scripts/repro/figure10.sh:
  <output_path>/<model>/mi/attn_sink_base_models_gen_neg_acc.pkl      per-layer correct counts, negative prompts
  <output_path>/<model>/mi/attn_sink_pos_control_sweep_pos_acc.pkl    per-layer correct counts, positive prompts
  <logs_dir>/f10_neg_sweep_<model>.log                                "Neg vanilla accuracy: X" (no-sink baseline)
  <logs_dir>/f10_pos_sweep_<model>.log                                "Pos vanilla accuracy: X"

    python src/zzj_mi/mi_module/analysis_module/viz_attn_sink_neg_pos_per_layer.py \
        --output_path outputs_release --logs_dir logs --figs_dir figs
Writes <figs_dir>/negation_circuit/causal_ablation/all_models_neg_pos_sweep.pdf
"""
import pickle
import re
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (display name, Hugging Face model id)
MODELS = [
    ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
    ("Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1"),
    ("OLMo-2-7B", "allenai/OLMo-2-1124-7B"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
    ("Gemma-2-9b", "google/gemma-2-9b"),
    ("Qwen3-8B-Base", "Qwen/Qwen3-8B-Base"),
]


def parse_args():
    p = ArgumentParser()
    p.add_argument("--output_path", default="outputs_release")
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--figs_dir", default="figs")
    p.add_argument("--total", type=int, default=648, help="rows in data/prompts-cleaned-release.json")
    return p.parse_args()


def read_vanilla(log_path: Path, label: str) -> float:
    """Return the last 'Neg|Pos vanilla accuracy: X' value printed by the sweep script."""
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} not found; run scripts/repro/figure10.sh first")
    hits = re.findall(rf"^{label} vanilla accuracy:\s*([0-9.]+)", log_path.read_text(), re.M)
    if not hits:
        raise ValueError(f"no '{label} vanilla accuracy' line in {log_path}")
    return float(hits[-1])


def load_model(model_id: str, output_path: Path, logs_dir: Path, total: int):
    slug = model_id.replace("/", "--")
    short = model_id.split("/")[-1]
    mi = output_path / slug / "mi"
    neg_counts = pickle.load(open(mi / "attn_sink_base_models_gen_neg_acc.pkl", "rb"))
    pos_counts = pickle.load(open(mi / "attn_sink_pos_control_sweep_pos_acc.pkl", "rb"))
    if len(neg_counts) != len(pos_counts):
        raise ValueError(f"{model_id}: neg/pos sweep lengths differ ({len(neg_counts)} vs {len(pos_counts)})")
    neg_acc = [c / total for c in neg_counts]
    pos_acc = [c / total for c in pos_counts]
    v_neg = read_vanilla(logs_dir / f"f10_neg_sweep_{short}.log", "Neg")
    v_pos = read_vanilla(logs_dir / f"f10_pos_sweep_{short}.log", "Pos")
    return neg_acc, pos_acc, v_neg, v_pos


def draw(ax, name, neg_acc, pos_acc, v_neg, v_pos):
    nl = len(neg_acc)
    xs = list(range(nl))
    neg_best = max(range(nl), key=lambda i: neg_acc[i])
    ax.plot(xs, neg_acc, marker="o", ms=3.5, lw=1.4, color="C0", label="neg acc (sink)")
    ax.plot(xs, pos_acc, marker="s", ms=3.5, lw=1.4, color="C1", label="pos acc (sink)")
    ax.axhline(v_neg, color="C0", ls=":", lw=1.0, alpha=0.8, label=f"vanilla neg = {v_neg:.3f}")
    ax.axhline(v_pos, color="C1", ls=":", lw=1.0, alpha=0.8, label=f"vanilla pos = {v_pos:.3f}")
    ax.axvline(neg_best, color="green", ls="--", lw=1.2, label=f"argmax neg 0-idx={neg_best})")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(f"{name}  (L={nl})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    print(f"{name}: vanilla neg {v_neg:.4f} pos {v_pos:.4f}; best sink neg acc {neg_acc[neg_best]:.4f} "
          f"at layer {neg_best} (0-indexed; 1-indexed in the paper's Table 9)")


def main():
    args = parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for ax, (name, model_id) in zip(axes.flat, MODELS):
        draw(ax, name, *load_model(model_id, Path(args.output_path), Path(args.logs_dir), args.total))
    for ax in axes[-1]:
        ax.set_xlabel("Sink layer (0-indexed; sink from this layer onward)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    fig.tight_layout()
    out_pdf = Path(args.figs_dir) / "negation_circuit/causal_ablation" / "all_models_neg_pos_sweep.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved figure to: {out_pdf}")


if __name__ == "__main__":
    main()
