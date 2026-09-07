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
src/zzj_mi/mi_module/analysis_module/inspect_patch_sink_multil_sweepl_expanded.py

Paper Figure 4 (truncated sink plot, Llama), Figure 6 (Llama) and Figure 7
(Mistral): windowed attention-sink ablation and attention-output path patching,
negative accuracy vs. window center layer, window sizes 1-5. Reads
  <output_path>/<model>/mi/attn_sink_ablation_multil_sweepl_w{1..5}_fix_scores_sweep.pt
  <output_path>/<model>/mi/patch_attn_multil_sweepl_w{1..5}_fix_scores_sweep.pt
(from attn_sink_ablation_multil_sweepl_fix_scores_expanded.py and
patch_attn_multil_sweepl_fix_scores_expanded.py, both run with --should_save true).

    python src/zzj_mi/mi_module/analysis_module/inspect_patch_sink_multil_sweepl_expanded.py \
        --model_path meta-llama/Llama-3.1-8B --output_path outputs_release --figs_dir figs
Writes attn_out_path_patching_window_<model>.pdf, attn_sink_ablation_window_<model>.pdf and
attn_sink_ablation_window_truncated_<model>.pdf under <figs_dir>/negation_circuit/causal_ablation/.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as t
from datasets import load_dataset


WINDOW_SIZES = [1, 2, 3, 4, 5]


DEFAULT_FIGS_DIR = "figs"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data_path", default="data/prompts-cleaned-multi-release.json")
    parser.add_argument("--output_path", default="outputs_release")
    parser.add_argument("--output_subdir", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--start_layer", type=int, default=8)
    parser.add_argument("--figs_dir", default=DEFAULT_FIGS_DIR,
                        help="Figure root; PDFs land in <figs_dir>/negation_circuit/causal_ablation/")
    return parser.parse_args()


def build_results_dir(model_path: str, output_path: str, output_subdir: str | None) -> Path:
    base_dir = Path(output_path) / model_path.replace("/", "--") / "mi"
    if output_subdir:
        return base_dir / output_subdir
    return base_dir


def load_windowed_results(results_dir: Path, prefix: str):
    window_data = {}
    for w_size in WINDOW_SIZES:
        file_path = results_dir / f"{prefix}_w{w_size}_fix_scores_sweep.pt"
        data = t.load(file_path)
        window_data[w_size] = data
        print(f"\n{prefix} - Window size {w_size}:")
        print(f"  File: {file_path}")
        print(f"  Keys: {data.keys()}")
        print(f"  n_layers: {data['n_layers']}")
        print(f"  Accuracy configs: {len(data['accuracy_per_config'])}")
    return window_data


def main():
    args = parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else build_results_dir(
        args.model_path,
        args.output_path,
        args.output_subdir,
    )
    print(f"Results directory: {results_dir}")

    full_data = load_dataset("json", data_files=args.data_path, split="train")
    full_num_entries = full_data.num_rows
    print(f"Loaded {full_num_entries} examples from {args.data_path}")

    patch_window_data = load_windowed_results(
        results_dir,
        "patch_attn_multil_sweepl",
    )
    sink_window_data = load_windowed_results(
        results_dir,
        "attn_sink_ablation_multil_sweepl",
    )

    n_layers = patch_window_data[1]["n_layers"]
    print(f"\nModel has {n_layers} layers")

    figures_dir = Path(args.figs_dir) / "negation_circuit/causal_ablation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model_path.replace("/", "--")

    colors = ["steelblue", "coral", "mediumseagreen", "purple", "crimson"]

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    for w_size, color in zip(WINDOW_SIZES, colors):
        data = patch_window_data[w_size]
        acc = data["accuracy_per_config"][:-1]
        control_acc = data["accuracy_per_config"][-1]
        x = [i + (w_size - 1) / 2 for i in range(len(acc))]

        ax1.plot(
            x,
            [1 - a for a in acc],
            "o-",
            linewidth=2,
            label=f"Window Size {w_size}",
            color=color,
            markersize=4,
            alpha=0.8,
        )
        ax1.axhline(y=1 - control_acc, color=color, linestyle="--", linewidth=1, alpha=0.3)

    ax1.set_xlabel("Center Layer", fontsize=12)
    ax1.set_ylabel("Neg Acc", fontsize=12)
    ax1.set_title(f"Attention Output Path Patching - {args.model_path}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, n_layers, 4))
    plt.tight_layout()
    _patch_path = figures_dir / f"attn_out_path_patching_window_{model_slug}.pdf"
    plt.savefig(_patch_path)
    print(f"✓ Saved figure to: {_patch_path}")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))
    for w_size, color in zip(WINDOW_SIZES, colors):
        data = sink_window_data[w_size]
        acc = data["accuracy_per_config"][:-1]
        control_acc = data["accuracy_per_config"][-1]
        x = [i + (w_size - 1) / 2 for i in range(len(acc))]

        ax1.plot(
            x,
            acc,
            "o-",
            linewidth=2,
            label=f"Window Size {w_size}",
            color=color,
            markersize=4,
            alpha=0.8,
        )
        ax1.axhline(y=control_acc, color=color, linestyle="--", linewidth=1, alpha=0.3)

    ax1.set_xlabel("Center Layer", fontsize=12)
    ax1.set_ylabel("Neg Acc", fontsize=12)
    ax1.set_title(f"Attention Sink Ablation - {args.model_path}", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, n_layers, 4))
    plt.tight_layout()
    _sink_path = figures_dir / f"attn_sink_ablation_window_{model_slug}.pdf"
    plt.savefig(_sink_path)
    print(f"✓ Saved figure to: {_sink_path}")

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    plt.rcParams.update({"font.size": 16})

    for w_size, color in zip(WINDOW_SIZES, colors):
        data = sink_window_data[w_size]
        acc_all = data["accuracy_per_config"][:-1]
        control_acc = data["accuracy_per_config"][-1]
        acc = acc_all[args.start_layer:]
        x = [args.start_layer + i + (w_size - 1) / 2 for i in range(len(acc))]

        ax1.plot(
            x,
            acc,
            "o-",
            linewidth=2,
            label=f"Window Size {w_size}",
            color=color,
            markersize=4,
            alpha=0.8,
        )
        ax1.axhline(y=control_acc, color=color, linestyle="--", linewidth=1, alpha=0.3)

    ax1.set_xlabel("Center Layer")
    ax1.set_ylabel("Neg Acc")
    ax1.legend(fontsize=12, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=args.start_layer - 0.5, right=n_layers - 0.5)
    ax1.set_xticks(range(args.start_layer, n_layers, 4))
    plt.tight_layout()
    _sink_trunc_path = figures_dir / f"attn_sink_ablation_window_truncated_{model_slug}.pdf"
    plt.savefig(_sink_trunc_path)
    print(f"✓ Saved figure to: {_sink_trunc_path}")


if __name__ == "__main__":
    main()
