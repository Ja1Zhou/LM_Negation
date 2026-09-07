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
src/zzj_mi/mi_module/analysis_module/inspect_last_not_B_hidden_expanded.py

Paper Figure 3 (PCA of residual-stream hidden states at one component, with
neg->pos arrows for sampled pairs) and Figure 11 (2D PCA for a range of
components). Reads the cache written by save_last_not_B_hidden_multi_template.py:
  <output_path>/<model>/mi/second_last_hidden_multi_template.pt
    positive / negative: [num_components, n_examples, d_model]
    resid_labels: component names, ordered "0_pre, 0_mid, 0_post, 1_mid, 1_post, ..."
    (index 2k+1 = layer k after attention = "k_mid"; index 2k+2 = layer k after MLP = "k_post")

    python src/zzj_mi/mi_module/analysis_module/inspect_last_not_B_hidden_expanded.py \
        --model_path meta-llama/Llama-3.1-8B --output_path outputs_release --figs_dir figs
Writes <figs_dir>/negation_circuit/PCA/<model>-<label>.pdf  (Figure 3; --component 23 = "11_mid")
   and <figs_dir>/negation_circuit/PCA/pca_2d.pdf             (Figure 11; --subset_start/--subset_end)
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.decomposition import PCA


def parse_args():
    p = ArgumentParser()
    p.add_argument("--model_path", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--output_path", default="outputs_release")
    p.add_argument("--figs_dir", default="figs")
    p.add_argument("--component", type=int, default=23, help="component index for the single-panel figure (23 = 11_mid)")
    p.add_argument("--subset_start", type=int, default=20)
    p.add_argument("--subset_end", type=int, default=32, help="exclusive")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fit_pca(pos: np.ndarray, neg: np.ndarray):
    X = np.concatenate([pos, neg], axis=0)
    y = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])  # 1=pos, 0=neg
    pca = PCA(n_components=3)
    Z = pca.fit_transform(X)
    return Z, y, pca.explained_variance_ratio_


def draw_example_pair_arrows(ax, Z, sampled_indices, n_examples):
    Z_pos, Z_neg = Z[:n_examples, :2], Z[n_examples:, :2]  # concatenation order: pos then neg
    for idx in sampled_indices:
        start, end = Z_pos[idx], Z_neg[idx]
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=0.15, head_length=0.1,
                 fc="green", ec="green", alpha=0.75, linewidth=1.5, length_includes_head=True, zorder=10)


def main():
    args = parse_args()
    slug = args.model_path.replace("/", "--")
    pt_file = Path(args.output_path) / slug / "mi" / "second_last_hidden_multi_template.pt"
    print(f"Loading from: {pt_file}")
    data = t.load(pt_file)
    pos_all = data["positive"].float().cpu()
    neg_all = data["negative"].float().cpu()
    resid_labels = data.get("resid_labels", None)
    num_components, n_examples, d_model = pos_all.shape
    print(f"components={num_components}, examples={n_examples}, d_model={d_model}")
    labels = list(resid_labels) if resid_labels is not None else [f"component_{i}" for i in range(num_components)]

    pca = {i: fit_pca(pos_all[i].numpy(), neg_all[i].numpy()) for i in range(num_components)}
    figs = Path(args.figs_dir) / "negation_circuit/PCA"
    figs.mkdir(parents=True, exist_ok=True)

    # ---- Figure 11: subset grid (3 x 4), 3 sampled pairs with arrows ----
    rng = np.random.RandomState(args.seed)
    sampled = rng.choice(n_examples, 3, replace=False)
    comps = list(range(args.subset_start, args.subset_end))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), squeeze=False)
    for k, comp in enumerate(comps):
        ax = axes[k // 4][k % 4]
        Z, y, _ = pca[comp]
        ax.scatter(Z[y == 0, 0], Z[y == 0, 1], c="crimson", s=18, alpha=0.6)
        ax.scatter(Z[y == 1, 0], Z[y == 1, 1], c="royalblue", s=18, alpha=0.6)
        draw_example_pair_arrows(ax, Z, sampled, n_examples)
        ax.set_title(labels[comp], fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.15)
    fig.suptitle(f"PCA Overview (2D) — Layers {args.subset_start // 2}-{args.subset_end // 2 - 1} — {args.model_path}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.6, w_pad=0.8, h_pad=0.8)
    out = figs / "pca_2d.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {out}")

    # ---- Figure 3: single component, 8 sampled pairs with arrows ----
    comp = args.component
    Z, y, evr = pca[comp]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    plt.rcParams.update({"font.size": 16})
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], c="crimson", alpha=0.6, s=20, label="Negative")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], c="royalblue", alpha=0.6, s=20, label="Positive")
    sampled8 = np.random.RandomState(args.seed).choice(n_examples, 8, replace=False)
    draw_example_pair_arrows(ax, Z, sampled8, n_examples)
    ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = figs / f"{slug}-{labels[comp]}.pdf"
    plt.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {out}  (component {comp} = {labels[comp]}; PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%)")


if __name__ == "__main__":
    main()
