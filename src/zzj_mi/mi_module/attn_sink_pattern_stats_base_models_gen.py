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
Attention mass on the sink token set (paper Table 1).

For every negative prompt, the post-softmax attention pattern of the last
token is read at every layer and head, and the mass placed on the first
token plus the mass placed on the token itself is summed. The script prints
the average over all layers, heads and prompts ("Overall average sink-set
attention mass").

    python src/zzj_mi/mi_module/attn_sink_pattern_stats_base_models_gen.py \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-release.json \
        --batch_size 648 --should_save false
"""
from pathlib import Path
from functools import partial
from typing import cast

import torch as t
from datasets import load_dataset
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerFast

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_gen import DataProcessor
from tqdm import tqdm

CUR_FILE_STEM = Path(__file__).stem
DEFAULT_FLOAT_DTYPE = t.bfloat16


def get_first_token_pos(attention_mask: Int[t.Tensor, "b seq"]) -> Int[t.Tensor, "b"]:
    """
    For left-padded inputs, the first real token is the first index where the mask is 1.
    """
    return attention_mask.argmax(dim=-1)


def is_pattern_hook(hook_name: str) -> bool:
    return hook_name.endswith("hook_pattern")


def accumulate_last_token_attention_stats(
    pattern: Float[t.Tensor, "b h q k"],
    hook: HookPoint,
    first_token_pos: Int[t.Tensor, "b"],
    first_token_sums: Float[t.Tensor, "layer head"],
    self_token_sums: Float[t.Tensor, "layer head"],
    sink_token_sums: Float[t.Tensor, "layer head"],
):
    """
    Accumulate attention mass from the last query position onto:
    1. the first real token
    2. the token itself
    3. the union of those two positions
    """
    layer = hook.layer()
    _, num_heads, _, seq_len = pattern.shape

    last_query_pattern = pattern[:, :, -1, :].to(t.float32)  # [b, h, k]

    first_token_idx = first_token_pos[:, None, None].expand(-1, num_heads, 1)
    first_token_mass = last_query_pattern.gather(dim=-1, index=first_token_idx).squeeze(-1)
    self_token_mass = last_query_pattern[:, :, -1]

    sink_token_mass = self_token_mass.clone()
    distinct_first_mask = (first_token_pos != (seq_len - 1)).to(t.float32).unsqueeze(-1)
    sink_token_mass += first_token_mass * distinct_first_mask

    first_token_sums[layer] += first_token_mass.sum(dim=0).cpu().to(t.float64)
    self_token_sums[layer] += self_token_mass.sum(dim=0).cpu().to(t.float64)
    sink_token_sums[layer] += sink_token_mass.sum(dim=0).cpu().to(t.float64)

    return pattern


if __name__ == "__main__":
    t.set_grad_enabled(False)

    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"

    full_data = load_dataset("json", data_files=args.data_path, split="train")
    full_num_entries = full_data.num_rows
    print(f"Loaded {full_num_entries} examples")
    print("Reporting attention statistics on negative prompts only")

    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=DEFAULT_FLOAT_DTYPE,
        default_padding_side="left",
        device=device,
    )

    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True

    first_token_sums = t.zeros(model.cfg.n_layers, model.cfg.n_heads, dtype=t.float64)
    self_token_sums = t.zeros_like(first_token_sums)
    sink_token_sums = t.zeros_like(first_token_sums)

    processed_num_entries = 0
    total_num_batches = (full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0)

    for data in tqdm(full_data.iter(batch_size=args.batch_size), total=total_num_batches):
        batch_size = len(data["negative_q"])
        processed_num_entries += batch_size

        _, neg_inputs, _ = data_processor.get_model_inputs(data, tok)
        neg_first_token_pos = get_first_token_pos(neg_inputs["attention_mask"])

        stats_hook = partial(
            accumulate_last_token_attention_stats,
            first_token_pos=neg_first_token_pos,
            first_token_sums=first_token_sums,
            self_token_sums=self_token_sums,
            sink_token_sums=sink_token_sums,
        )

        model.run_with_hooks(
            input=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
            return_type=None,
            fwd_hooks=[(is_pattern_hook, stats_hook)],
        )

    if processed_num_entries != full_num_entries:
        print(
            f"WARNING: processed {processed_num_entries} examples, expected {full_num_entries}"
        )

    first_token_mean = first_token_sums / processed_num_entries
    self_token_mean = self_token_sums / processed_num_entries
    sink_token_mean = sink_token_sums / processed_num_entries

    layer_first_token_mean = first_token_mean.mean(dim=-1)
    layer_self_token_mean = self_token_mean.mean(dim=-1)
    layer_sink_token_mean = sink_token_mean.mean(dim=-1)

    overall_first_token_mean = first_token_mean.mean().item()
    overall_self_token_mean = self_token_mean.mean().item()
    overall_sink_token_mean = sink_token_mean.mean().item()

    print()
    print("Final results:")
    print(f"Total valid samples: {processed_num_entries}")
    print(f"Overall average first-token attention mass: {overall_first_token_mean:.4f}")
    print(f"Overall average self-token attention mass: {overall_self_token_mean:.4f}")
    print(f"Overall average sink-set attention mass: {overall_sink_token_mean:.4f}")
    print()
    print("Per-layer averages over heads:")
    for layer in range(model.cfg.n_layers):
        print(
            f"Layer {layer:02d}: "
            f"first={layer_first_token_mean[layer].item():.4f}, "
            f"self={layer_self_token_mean[layer].item():.4f}, "
            f"sink={layer_sink_token_mean[layer].item():.4f}"
        )
    print()

    max_layer_sink = layer_sink_token_mean.argmax().item()
    print(
        f"Max per-layer sink average: {layer_sink_token_mean[max_layer_sink].item():.4f} "
        f"at layer {max_layer_sink}"
    )

    if args.should_save:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{CUR_FILE_STEM}.pt"

        output_obj = {
            "prompt_type": "negative_q",
            "num_samples": processed_num_entries,
            "first_token_sums": first_token_sums,
            "self_token_sums": self_token_sums,
            "sink_token_sums": sink_token_sums,
            "first_token_mean": first_token_mean,
            "self_token_mean": self_token_mean,
            "sink_token_mean": sink_token_mean,
            "layer_first_token_mean": layer_first_token_mean,
            "layer_self_token_mean": layer_self_token_mean,
            "layer_sink_token_mean": layer_sink_token_mean,
            "overall_first_token_mean": overall_first_token_mean,
            "overall_self_token_mean": overall_self_token_mean,
            "overall_sink_token_mean": overall_sink_token_mean,
        }
        t.save(output_obj, output_file)

        print()
        print(f"Saved results to: {output_file}")
        print("Saved data includes:")
        print(f"  - first_token_mean: shape {tuple(first_token_mean.shape)}")
        print(f"  - self_token_mean: shape {tuple(self_token_mean.shape)}")
        print(f"  - sink_token_mean: shape {tuple(sink_token_mean.shape)}")
