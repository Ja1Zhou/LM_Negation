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
src/zzj_mi/mi_module/analysis_module/inspect_last_not_B_hidden_separability_cross_valid_max.py

Paper Figure 8: cross-validated accuracy of decoding "not" from the residual
stream as a function of component index (Sec. 5.1 "Experiment Pipeline").
10-fold CV over example pairs; per fold, PCA(2)+LDA is fit at every candidate
anchor component on the train split, the anchor with the highest TRAIN accuracy
is kept, and its direction (with a midpoint threshold) is evaluated at every
component on the held-out split. Reads the same cache as Figure 3:
  <output_path>/<model>/mi/second_last_hidden_multi_template.pt

    python src/zzj_mi/mi_module/analysis_module/inspect_last_not_B_hidden_separability_cross_valid_max.py \
        --model_path meta-llama/Llama-3.1-8B --output_path outputs_release --figs_dir figs
Writes <figs_dir>/negation_circuit/causal_ablation/decode_not_cv_<model>.pdf
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold


def parse_args():
    p = ArgumentParser()
    p.add_argument("--model_path", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--output_path", default="outputs_release")
    p.add_argument("--figs_dir", default="figs")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--random_state", type=int, default=0)
    return p.parse_args()


def fit_anchor_direction_pca_lda(pos_anchor: t.Tensor, neg_anchor: t.Tensor) -> np.ndarray:
    X_tr = t.cat([pos_anchor, neg_anchor], dim=0).numpy()
    y_tr = np.concatenate([np.ones(pos_anchor.shape[0], dtype=int), np.zeros(neg_anchor.shape[0], dtype=int)])
    pca = PCA(n_components=2, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_tr_pca, y_tr)
    w_orig = pca.components_.T @ lda.coef_.reshape(-1)
    w_orig = w_orig / (np.linalg.norm(w_orig) + 1e-12)
    return w_orig.astype(np.float32)


def midpoint(z_pos_tr, z_neg_tr):
    return 0.5 * (float(z_pos_tr.mean()) + float(z_neg_tr.mean()))


def threshold_acc(z_pos, z_neg, t_mid):
    return 0.5 * ((z_pos > t_mid).mean() + (z_neg <= t_mid).mean())


def cv_acc_max_anchor(pos_all, neg_all, labels, n_splits, random_state):
    C, N, _ = pos_all.shape
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_all = np.zeros((n_splits, C), dtype=np.float32)
    chosen = np.zeros(n_splits, dtype=np.int32)
    for fold_id, (tr, va) in enumerate(kf.split(np.arange(N))):
        best_c, best_train_acc, best_w = -1, -1.0, None
        for c in range(C):
            w = fit_anchor_direction_pca_lda(pos_all[c, tr], neg_all[c, tr])
            w_t = t.from_numpy(w).to(pos_all.dtype)
            z_pos_tr, z_neg_tr = (pos_all[c, tr] @ w_t).numpy(), (neg_all[c, tr] @ w_t).numpy()
            train_acc = threshold_acc(z_pos_tr, z_neg_tr, midpoint(z_pos_tr, z_neg_tr))
            if train_acc > best_train_acc:
                best_c, best_train_acc, best_w = c, train_acc, w
        chosen[fold_id] = best_c
        w_t = t.from_numpy(best_w).to(pos_all.dtype)
        for c in range(C):
            z_pos_tr, z_neg_tr = (pos_all[c, tr] @ w_t).numpy(), (neg_all[c, tr] @ w_t).numpy()
            t_mid = midpoint(z_pos_tr, z_neg_tr)
            acc_all[fold_id, c] = threshold_acc((pos_all[c, va] @ w_t).numpy(), (neg_all[c, va] @ w_t).numpy(), t_mid)
        print(f"Fold {fold_id + 1:2d}/{n_splits}: anchor={best_c} ({labels[best_c]}) "
              f"train_acc={best_train_acc:.3f} mean_val_acc={acc_all[fold_id].mean():.3f}")
    return acc_all.mean(0), acc_all.std(0), chosen


def main():
    args = parse_args()
    slug = args.model_path.replace("/", "--")
    pt_file = Path(args.output_path) / slug / "mi" / "second_last_hidden_multi_template.pt"
    print(f"Loading from: {pt_file}")
    data = t.load(pt_file)
    pos_all = data["positive"].float().cpu()
    neg_all = data["negative"].float().cpu()
    C, N, D = pos_all.shape
    labels = list(data["resid_labels"]) if data.get("resid_labels") is not None else [f"c{i}" for i in range(C)]
    print(f"components={C}, examples={N}, d_model={D}")

    acc_mean, acc_std, chosen = cv_acc_max_anchor(pos_all, neg_all, labels, args.n_splits, args.random_state)
    uniq, counts = np.unique(chosen, return_counts=True)
    print("Chosen anchors across folds: " + ", ".join(f"{labels[c]} x{n}" for c, n in zip(uniq, counts)))
    print(f"Best component: {labels[int(acc_mean.argmax())]} acc={acc_mean.max():.4f}")

    idx = np.arange(C)
    step = max(1, C // 20)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, acc_mean, linewidth=1.6, alpha=0.9, label="CV Acc")
    ax.scatter(idx, acc_mean, s=16, alpha=0.75)
    ax.errorbar(idx, acc_mean, yerr=acc_std, fmt="none", capsize=2, alpha=0.35)
    ax.set_xticks(idx[::step])
    ax.set_xticklabels([labels[i] for i in idx[::step]], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cross Validation Accuracy", fontsize=12)
    ax.set_xlabel("Residual Stream Component", fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"{args.model_path} (max-anchor selection)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(args.figs_dir) / "negation_circuit/causal_ablation" / f"decode_not_cv_{slug}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print(f"Saved figure to: {out}")


if __name__ == "__main__":
    main()
