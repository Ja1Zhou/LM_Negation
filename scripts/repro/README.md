# Reproduction scripts

Every script runs from the repo root inside the environment from `scripts/env_runbook.sh`
(`source scripts/setup_env.sh`), with the six base models already in the Hugging Face cache
(`HF_HUB_CACHE`) and, for `figure2.sh`, the 28 OLMo-2 checkpoint revisions listed in
`src/zzj_mi/olmo_module/checkpoints.txt`. Each script writes its logs to `logs/` and prints
or saves the paper item it reproduces.

| script | paper item | compute scripts |
|---|---|---|
| `table1.sh` | Table 1 `tab:attn_sink_mass` | `attn_sink_pattern_stats_base_models_gen.py` |
| `table2.sh` | Table 2 `tab:model_accuracies` | `logits_base_models_gen.py` |
| `table3.sh` | Table 3 `tab:neg_acc_comparison` + Table 9 `tab:best_layers` | `attn_sink_base_models_gen.py`, `logitlens_base_gen.py` |
| `table8.sh` | Table 8 `tab:multi_ans` | `logits_base_models_gen_expanded.py` |
| `figure2.sh` | Figure 2 `fig:olmo_train_accs` | `olmo_module/acc_n_sensitivity.py` + `viz_acc_n_sensitivity.py` |
| `figure3.sh` | Figures 3, 8, 11 | `save_last_not_B_hidden_multi_template.py` + PCA / LDA viz |
| `figure4.sh` | Figures 4, 6, 7 | `attn_sink_ablation_multil_sweepl_fix_scores_expanded.py`, `patch_attn_multil_sweepl_fix_scores_expanded.py` + viz |
| `figure5.sh` | Figures 5, 9 | `save_attn_outputs_{topk,leastk}_expanded.py` + OpenRouter annotation (`src/zzj_mi/annotate/`) + viz |
| `figure10.sh` | Figure 10 `fig:attn_sink_neg_pos_sweep` | `attn_sink_base_models_gen.py --should_save true`, `attn_sink_pos_control_sweep_base_models_gen.py` + viz |

Every script shards its models round-robin over the GPUs `nvidia-smi` sees (one queue per GPU), so it runs on 1 GPU (serially) up to 6-8 GPUs (fully parallel).
