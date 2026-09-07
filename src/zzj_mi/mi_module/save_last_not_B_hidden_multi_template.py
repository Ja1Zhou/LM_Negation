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
Cache residual-stream hidden states at the "last token of not Y" position
(paper Figures 3, 8 and 11).

For every prompt pair and every residual-stream component (embedding, then
after-attention and after-MLP at each layer) the hidden state at the target
position is saved, for both the positive and the negative prompt. The target
position depends only on the template (see data_module/multi_template_target.py):
stem / something_that / article -> second-to-last token, question -> fourth-to-last.

Output <output_path>/<model>/mi/second_last_hidden_multi_template.pt:
    'positive', 'negative':  [2*n_layers+1, n_examples, d_model]
    'resid_labels':          component names ("0_pre", "0_mid", "0_post", "1_mid", ...)
    'template_ids', 'prompt_numbers', 'n_layers', 'd_model', 'n_examples'
Examples are stored in template-block order (stem, something_that, article,
question; 162 each), the order of the input file.

    python src/zzj_mi/mi_module/save_last_not_B_hidden_multi_template.py \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \
        --output_path outputs_release/ --batch_size 162 --should_save true
"""

from __future__ import annotations

from functools import partial
from typing import cast

import torch as t
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerFast

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.multi_template_expanded import DataProcessor
from zzj_mi.mi_module.data_module.multi_template_target import (
    TEMPLATE_INDEX_TO_ID,
    N_STEMS_PER_TEMPLATE,
    infer_template_id,
)

DEFAULT_FLOAT_DTYPE = t.bfloat16

TEMPLATE_NEG_SLICE: dict[str, int] = {
    "stem":           -2,
    "something_that": -2,
    "article":        -2,
    "question":       -4,
}


def filter_resid_components(name: str, n_layers: int) -> bool:
    return (
        name.endswith("resid_pre")
        or name.endswith("resid_mid")
        or name == get_act_name("resid_post", n_layers - 1)
    )


def main():
    t.set_grad_enabled(False)

    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)

    device = (
        "cuda"
        if t.cuda.is_available()
        else "mps"
        if t.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    full_data = load_dataset("json", data_files=args.data_path, split="train")
    full_num_entries = full_data.num_rows
    print(f"Loaded {full_num_entries} examples from {args.data_path}\n")

    if "template_id" in full_data.column_names:
        all_template_ids = list(full_data["template_id"])
    elif "prompt_number" in full_data.column_names:
        all_template_ids = [infer_template_id(int(p)) for p in full_data["prompt_number"]]
    else:
        raise ValueError(
            "Data file must have either 'template_id' or 'prompt_number' on every row."
        )
    all_prompt_numbers = (
        [int(p) for p in full_data["prompt_number"]]
        if "prompt_number" in full_data.column_names
        else None
    )

    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
        print(f"Set pad_token to eos_token: {tok.eos_token}\n")
    if not tok.is_fast:
        raise RuntimeError("Multi-template pipeline requires a fast tokenizer.")

    print(f"Loading model: {args.model_path}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=DEFAULT_FLOAT_DTYPE,
        default_padding_side="left",
        trust_remote_code=True,
    )
    model.eval()
    cache_filter = partial(filter_resid_components, n_layers=model.cfg.n_layers)
    print(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}\n")

    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True

    if not args.should_save:
        print("WARNING: --should_save not set, outputs will NOT be saved!\n")
        output_file = None
    else:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "second_last_hidden_multi_template.pt"
        print(f"Output file: {output_file}\n")

    print(f"Will cache {2 * model.cfg.n_layers + 1} residual stream activations per prompt")
    print("Components: resid_pre and resid_mid for each layer, plus final resid_post\n")

    # Group rows by template so we can apply a fixed pos_slice per template.
    template_indices: dict[str, list[int]] = {tpl: [] for tpl in TEMPLATE_INDEX_TO_ID}
    for i, tid in enumerate(all_template_ids):
        template_indices[tid].append(i)
    for tpl, idxs in template_indices.items():
        if len(idxs) != N_STEMS_PER_TEMPLATE:
            print(
                f"  [warn] template {tpl!r} has {len(idxs)} rows "
                f"(expected {N_STEMS_PER_TEMPLATE})"
            )

    all_pos_hidden: list[t.Tensor] = []
    all_neg_hidden: list[t.Tensor] = []
    out_template_ids: list[str] = []
    out_prompt_numbers: list[int] | None = [] if all_prompt_numbers is not None else None
    pos_labels: list[str] | None = None

    has_logged_sample = False
    with tqdm(total=full_num_entries, desc="Processing examples") as pbar:
        for tpl in TEMPLATE_INDEX_TO_ID:
            row_ids = template_indices[tpl]
            if not row_ids:
                continue
            tpl_data = full_data.select(row_ids)
            pos_slice = TEMPLATE_NEG_SLICE[tpl]
            print(f"\n--- template={tpl}  rows={len(row_ids)}  pos_slice={pos_slice} ---")

            for data in tpl_data.iter(batch_size=args.batch_size):
                batch_size = len(data["positive_q"])
                pos_inputs, neg_inputs = data_processor.get_prompt_inputs(data, tok)

                _, pos_cache = model.run_with_cache(
                    input=pos_inputs["input_ids"],
                    attention_mask=pos_inputs["attention_mask"],
                    pos_slice=pos_slice,
                    names_filter=cache_filter,
                    return_type=None,
                )
                pos_hidden, batch_pos_labels = pos_cache.accumulated_resid(
                    incl_mid=True, return_labels=True
                )
                pos_hidden = pos_hidden.squeeze(-2)  # [2*n_layers+1, batch, d_model]
                del pos_cache

                _, neg_cache = model.run_with_cache(
                    input=neg_inputs["input_ids"],
                    attention_mask=neg_inputs["attention_mask"],
                    pos_slice=pos_slice,
                    names_filter=cache_filter,
                    return_type=None,
                )
                neg_hidden = neg_cache.accumulated_resid(incl_mid=True).squeeze(-2)
                del neg_cache

                if pos_labels is None:
                    pos_labels = batch_pos_labels

                if not has_logged_sample:
                    has_logged_sample = True
                    print("\n=== First batch sample ===")
                    print(f"Template: {tpl}, pos_slice={pos_slice}")
                    print(f"Batch size: {batch_size}")
                    print(f"Positive hidden states shape: {pos_hidden.shape}")
                    print(f"Negative hidden states shape: {neg_hidden.shape}")
                    print(f"Resid labels: {pos_labels}")
                    print(f"Example pos prompt: {data['positive_q'][0]}")
                    print(f"Example neg prompt: {data['negative_q'][0]}")
                    pos_tok_str = tok.convert_ids_to_tokens(
                        [pos_inputs["input_ids"][0, pos_slice].item()]
                    )[0]
                    neg_tok_str = tok.convert_ids_to_tokens(
                        [neg_inputs["input_ids"][0, pos_slice].item()]
                    )[0]
                    print(f"Pos sliced token (idx={pos_slice}): {pos_tok_str!r}")
                    print(f"Neg sliced token (idx={pos_slice}): {neg_tok_str!r}\n")
                    pos_norms = t.norm(pos_hidden[:, 0, :], dim=-1)
                    neg_norms = t.norm(neg_hidden[:, 0, :], dim=-1)
                    print("L2 norms across components (first example):")
                    print(
                        f"  Positive: min={pos_norms.min():.2f}, max={pos_norms.max():.2f}, "
                        f"mean={pos_norms.mean():.2f}"
                    )
                    print(
                        f"  Negative: min={neg_norms.min():.2f}, max={neg_norms.max():.2f}, "
                        f"mean={neg_norms.mean():.2f}\n"
                    )

                all_pos_hidden.append(pos_hidden)
                all_neg_hidden.append(neg_hidden)
                out_template_ids.extend([tpl] * batch_size)
                if out_prompt_numbers is not None and "prompt_number" in data:
                    out_prompt_numbers.extend(int(p) for p in data["prompt_number"])
                pbar.update(batch_size)

    all_pos_hidden = t.cat(all_pos_hidden, dim=1)
    all_neg_hidden = t.cat(all_neg_hidden, dim=1)

    print("\n=== Complete ===")
    print(f"Processed {full_num_entries} examples")
    print(f"Final tensor shapes:")
    print(f"  Positive: {all_pos_hidden.shape}")
    print(f"  Negative: {all_neg_hidden.shape}")

    if args.should_save and output_file is not None:
        output_obj = {
            "positive": all_pos_hidden.cpu(),
            "negative": all_neg_hidden.cpu(),
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "n_examples": full_num_entries,
            "resid_labels": pos_labels,
            "template_ids": out_template_ids,
        }
        if out_prompt_numbers is not None:
            output_obj["prompt_numbers"] = out_prompt_numbers
        t.save(output_obj, output_file)
        print(f"\n✓ Saved to: {output_file}")
        print("\nTo load:")
        print(f"  data = torch.load('{output_file}')")
        print(
            f"  pos_hidden = data['positive']  # "
            f"[{2 * model.cfg.n_layers + 1}, {full_num_entries}, {model.cfg.d_model}]"
        )
        print(
            f"  neg_hidden = data['negative']  # "
            f"[{2 * model.cfg.n_layers + 1}, {full_num_entries}, {model.cfg.d_model}]"
        )
        print("  template_ids = data['template_ids']  # for per-template slicing")
    else:
        print("No outputs saved (use --should_save to save)")
    print()


if __name__ == "__main__":
    main()
