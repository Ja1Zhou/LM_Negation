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
Cumulative Attention Sink sweep on positive prompts (the positive curve of
paper Figure 10); positive-prompt counterpart of attn_sink_base_models_gen.py.

Prints the vanilla positive accuracy and, with --should_save true, pickles the
per-layer correct counts to
<output_path>/<model>/mi/attn_sink_pos_control_sweep_pos_acc.pkl
(pos_acc[i] = correct positive prompts when sinking from layer i onward).

    python src/zzj_mi/mi_module/attn_sink_pos_control_sweep_base_models_gen.py \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-release.json \
        --batch_size 648 --output_path outputs_release/ --should_save true
"""
import pickle
import re
import torch as t
from functools import partial
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    HfArgumentParser,
)
from typing import cast
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, Float

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_gen import DataProcessor
from tqdm import tqdm

DEFAULT_FLOAT_DTYPE = t.bfloat16

_HOOK_RE = re.compile(r"^blocks\.(\d+)\..*hook_attn_scores$")


def sink_attn_after_layer(hook_name: str, start_layer: int):
    m = _HOOK_RE.match(hook_name)
    if m is None:
        return False
    return int(m.group(1)) >= start_layer


def last_pos_layer_attn_sink_patch(
    attn_scores: Float[t.Tensor, "b h q k"],
    hook: HookPoint,
    bos_pos: Int[t.Tensor, "b"],
):
    b, h, q, k = attn_scores.shape
    last_q_scores = attn_scores[:, :, -1, :].clone()
    last_q_scores.fill_(t.finfo(attn_scores.dtype).min)
    allowed_pos = t.stack((t.full_like(bos_pos, -1), bos_pos), dim=0)
    last_q_scores[t.arange(b), :, allowed_pos] = attn_scores[t.arange(b), :, -1, allowed_pos]
    attn_scores[:, :, -1, :] = last_q_scores
    return attn_scores


if __name__ == "__main__":
    t.set_grad_enabled(False)

    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda"

    full_data = load_dataset("json", data_files=args.data_path, split="train")
    full_num_entries = full_data.num_rows
    print(f"Loaded {full_num_entries} examples")

    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=DEFAULT_FLOAT_DTYPE,
        default_padding_side="left",
    )

    # positive minus negative answer logits (positive ans is first in ans_ids)
    pos_minus_neg_logits = t.tensor([1, -1], dtype=DEFAULT_FLOAT_DTYPE).to(device)

    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True
    eps_threshold = 1e-6

    pos_vanilla_acc = 0
    pos_acc = [0 for _ in range(model.cfg.n_layers)]

    total_batches = (full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0)
    for data in tqdm(full_data.iter(batch_size=args.batch_size), total=total_batches):
        pos_inputs, _, ans_ids = data_processor.get_model_inputs(data, tok)

        # vanilla
        pos_logits = model(
            input=pos_inputs["input_ids"],
            attention_mask=pos_inputs["attention_mask"],
        )
        pos_logits = cast(Float[t.Tensor, "b seq vocab"], pos_logits)
        pos_logits_of_interest = pos_logits[:, -1, :].gather(dim=-1, index=ans_ids)
        pos_metrics = (pos_logits_of_interest * pos_minus_neg_logits).sum(dim=-1)
        pos_vanilla_acc += (pos_metrics > eps_threshold).sum().item()

        # bos position for the sink patch
        pos_attn_mask = pos_inputs["attention_mask"]
        pos_bos_pos = t.argmax(pos_attn_mask, dim=-1)
        patch_bos_pos_fn = partial(last_pos_layer_attn_sink_patch, bos_pos=pos_bos_pos)

        for attn_sink_layer in range(model.cfg.n_layers):
            filter_fn = partial(sink_attn_after_layer, start_layer=attn_sink_layer)
            pos_sink_logits = model.run_with_hooks(
                input=pos_inputs["input_ids"],
                attention_mask=pos_inputs["attention_mask"],
                fwd_hooks=[(filter_fn, patch_bos_pos_fn)],
            )
            pos_sink_logits = cast(Float[t.Tensor, "b seq vocab"], pos_sink_logits)
            pos_sink_logits_of_interest = pos_sink_logits[:, -1, :].gather(dim=-1, index=ans_ids)
            pos_sink_metrics = (pos_sink_logits_of_interest * pos_minus_neg_logits).sum(dim=-1)
            pos_acc[attn_sink_layer] += (pos_sink_metrics > eps_threshold).sum().item()

    print("Final results:")
    print(f"Total valid samples: {full_num_entries}")
    acc_per_layer = [pos_acc[layer] / full_num_entries for layer in range(model.cfg.n_layers)]
    print(f"Pos accuracies by layer (0-indexed): [{', '.join(f'{acc:.4f}' for acc in acc_per_layer)}]")
    max_acc = max(acc_per_layer)
    max_layer = acc_per_layer.index(max_acc)
    min_acc = min(acc_per_layer)
    min_layer = acc_per_layer.index(min_acc)
    print(f"Max pos sink accuracy: {max_acc:.4f} at layer {max_layer}")
    print(f"Min pos sink accuracy: {min_acc:.4f} at layer {min_layer}")
    print(f"Pos vanilla accuracy: {pos_vanilla_acc / full_num_entries:.4f}")
    print()

    if args.should_save:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "attn_sink_pos_control_sweep_pos_acc.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(pos_acc, f)
        print(f"\nSaved results to: {output_file}")
        print(f"  - pos_acc: list of length {len(pos_acc)}")
