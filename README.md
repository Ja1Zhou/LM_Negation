Code, data and run logs for the ICML 2026 paper *How Language Models Process Negation*.

## Layout

```
data/       prompts-cleaned-release.json        648 prompt pairs, one gold answer each
            prompts-cleaned-multi-release.json  same 648 pairs, multiple gold answers

src/zzj_mi/ arg_module/      shared CLI arguments (model, data, output paths, batch size)
            mi_module/       compute scripts (one per experiment) + data_module/ (prompt/answer tokenization)
            mi_module/analysis_module/   figure scripts
            olmo_module/     OLMo-2 checkpoint sweep
            annotate/        OpenRouter client + LLM-annotation script

scripts/    env_runbook.sh   exact environment build. setup_env.sh activates it
            repro/           one script per paper table/figure, see scripts/repro/README.md
            patches/         the one-line TransformerLens patch env_runbook.sh applies
```

## Setup

`scripts/env_runbook.sh` is the validated build (Python 3.12, torch 2.9.1+cu128, transformers 4.57.3,
TransformerLens from PR #816 at `9febc5cc` with `scripts/patches/transformer_lens_qwen3_base.patch`
so `Qwen/Qwen3-8B-Base` loads). Run its lines in order from the repo root with
[uv](https://docs.astral.sh/uv/) installed; torch must be installed first from the cu128 index or
pip resolves a different CUDA build. Afterwards `source scripts/setup_env.sh` activates the env.

Figure 2 needs the 28 OLMo-2 stage-1 revisions in `src/zzj_mi/olmo_module/checkpoints.txt`
(`generate_download_commands_from_checkpoints.py` prints the download commands, ~14 GB each).

## Reproducing the paper

Each table/figure has one script in `scripts/repro/` (mapping, GPU needs and runtimes in `scripts/repro/README.md`). From the repo root:

```bash
source scripts/setup_env.sh
bash scripts/repro/table2.sh     # Table 2: Neg/Pos accuracy + sensitivity, 6 models
bash scripts/repro/figure2.sh    # Figure 2: OLMo-2 accuracy across pre-training checkpoints
```

The scripts run one model per GPU in parallel; edit the `CUDA_VISIBLE_DEVICES` assignments to fit
fewer GPUs. Expected output: the tables are printed at the end of each script and the figures land in
`figs/`; `logs/` holds the run logs.

Figures 5 and 9 need an LLM annotator. The annotations used in the paper ship in
`outputs_release/<model>/mi/annotate_attn_output{,_suppression}_expanded/`, so `figure5.sh` makes no
API calls unless you delete them. To re-annotate, put `OPENROUTER_API_KEY=...` in the environment
(or pass `--dotenv path/to/.env`); the annotator (`src/zzj_mi/annotate/annotate_attn_outputs.py`) uses
`openai/gpt-oss-120b` at temperature 0, pinned to a bf16 endpoint, and caches every sample so
interrupted runs resume. 648 samples per model and mode cost a few cents.

## Data

`data/prompts-cleaned-release.json`: 162 (X, Y) concept pairs × 4 templates = 648 entries, each with a
positive prompt, a negative prompt and one single-token gold answer each. The multi-answer file adds up to
five gold answers per prompt.
The code is released under the Apache License 2.0 (`LICENSE`, attribution in `NOTICE`); the data under CC BY 4.0 (`data/LICENSE`). A `CITATION.cff` is included.

## Citation

```bibtex
@inproceedings{zhou2026-lmnegation,
  title={How Language Models Process Negation},
  author={Zhejian Zhou and Tianyi Zhou and Robin Jia and Jonathan May},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=8DLW34pkNi}
}
```