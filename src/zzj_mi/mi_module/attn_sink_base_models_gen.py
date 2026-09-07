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
Cumulative Attention Sink sweep on negative prompts (paper Table 3 "Attn. Sink"
column, Table 9, and the negative curve of Figure 10).

For each layer i, every attention head from layer i to the last layer is
restricted so the last token can attend only to the first token and to itself
("attention sink"); the negative accuracy is recorded per i, and the vanilla
(no-sink) accuracy is printed for reference. With --should_save true the
per-layer correct counts are pickled to
<output_path>/<model>/mi/attn_sink_base_models_gen_neg_acc.pkl.

    python src/zzj_mi/mi_module/attn_sink_base_models_gen.py \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-release.json \
        --batch_size 648 --should_save false
"""
# ###### Import packages ######
import pickle
import numpy as np
from functools import partial
import torch as t
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    HfArgumentParser,
)
from typing import cast
from datasets import load_dataset
from transformer_lens import (
    HookedTransformer,
)
from transformer_lens.utils import get_act_name
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, Float

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_gen import DataProcessor
from tqdm import tqdm

DEFAULT_FLOAT_DTYPE = t.bfloat16

# ###### Attention Sink patch functions ######
import re
_HOOK_RE = re.compile(r"^blocks\.(\d+)\..*hook_attn_scores$")

def sink_attn_after_layer(
    hook_name: str,
    start_layer: int,
):
    m = _HOOK_RE.match(hook_name)
    if m is None:
        return False
    layer = int(m.group(1))
    return layer >= start_layer

def last_pos_layer_attn_sink_patch(
    attn_scores: Float[t.Tensor, "b h q k"],
    hook: HookPoint,
    bos_pos: Int[t.Tensor, "b"],  # index of first non-pad token for each batch item
):
    """
    Before softmax, modify ONLY the last query row so it can attend to:
      1) itself (last key), and
      2) the first non-padding token (per-batch bos_pos).
    All other query rows are left unchanged.
    """
    b, h, q, k = attn_scores.shape

    last_q_scores = attn_scores[:, :, -1, :].clone() # [b, h, k]

    # Fill it with -inf, then selectively restore a couple entries
    last_q_scores.fill_(t.finfo(attn_scores.dtype).min)

    # 1) allow self (last key)
    # 2) allow first non-padding token per batch (bos_pos)
    allowed_pos = t.stack((t.full_like(bos_pos, -1), bos_pos), dim=0) # [2, b]
    last_q_scores[t.arange(b), :, allowed_pos] = attn_scores[t.arange(b), :, -1, allowed_pos]
    attn_scores[:, :, -1, :] = last_q_scores
    return attn_scores

if __name__ == "__main__":
    # we do not need gradients
    t.set_grad_enabled(False)
    # ###### Parse args ######
    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda"

    # ###### Load data ######
    # because we need to do a lot of batched operations we use datasets
    full_data = load_dataset("json", data_files=args.data_path, split="train")
    full_num_entries = full_data.num_rows
    print(f"Loaded {full_num_entries} examples")

    # load tokenizer
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # we need to pad, so set pad token
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    # ###### Load model ######
    """
    1. caveat of tl is that hard codes dtype to float32
    2. reminder to pad on the left side because we are reading say the last position logits
    """
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=DEFAULT_FLOAT_DTYPE, # checked all models
        default_padding_side="left", # this is just a reminder
    )

    """
    By default the first answer is positive and the second is negative
    - We want the logits difference between neg and pos
    """
    neg_minus_pos_logits = t.tensor([-1, 1], dtype=DEFAULT_FLOAT_DTYPE).to(device) # [2]

    # Initialize data processor
    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True
    eps_threshold = 1e-6

    # Track scores for neg vanilla
    neg_vanilla_acc = 0
    # Track scores for neg
    neg_acc = [0 for _ in range(model.cfg.n_layers)]

    # default plan is to use a batch size over here
    for batch_idx, data in enumerate(tqdm(full_data.iter(batch_size=args.batch_size), total=(full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0))):

        # ###### Calculate input ids and ans ids using DataProcessor ######
        pos_inputs, neg_inputs, ans_ids = data_processor.get_model_inputs(data, tok) # NOTE: pos ans first and then neg ans

        # first of all forward normally to get the logits difference
        neg_logits = model(
            input=neg_inputs['input_ids'],
            attention_mask=neg_inputs['attention_mask'],
        ) # neg_logits: [b, seq_len, vocab_size]
        # gather logits for the last position; in order to do this we need to build neg and pos ans tokens
        neg_logits = cast(Float[t.Tensor, "b seq_len vocab_size"], neg_logits)
        neg_logits_of_interest = neg_logits[:, -1, :].gather(dim=-1, index=ans_ids) # [b, 2]
        neg_metrics = (neg_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
        neg_vanilla_acc += (neg_metrics > eps_threshold).sum().item()
        
        # ###### Calculate bos position for attention sink patch ######
        """
        We need to calculate the position for the first token in the prompt for attention sink
        """
        neg_attn_mask = neg_inputs['attention_mask'] # [b, seq_len]
        neg_bos_pos = t.argmax(neg_attn_mask, dim=-1) # [b]
        patch_bos_pos_fn = partial(last_pos_layer_attn_sink_patch, bos_pos=neg_bos_pos)

        # ###### Calculate metrics ######
        for attn_sink_layer in range(model.cfg.n_layers):
            filter_fn = partial(sink_attn_after_layer, start_layer=attn_sink_layer)
            neg_logits = model.run_with_hooks(
                input=neg_inputs['input_ids'],
                attention_mask=neg_inputs['attention_mask'],
                fwd_hooks=[(filter_fn, patch_bos_pos_fn)],
            ) # neg_logits: [b, seq_len, vocab_size]

            # gather logits for the last position; in order to do this we need to build neg and pos ans tokens
            neg_logits = cast(Float[t.Tensor, "b seq_len vocab_size"], neg_logits)
            neg_logits_of_interest = neg_logits[:, -1, :].gather(dim=-1, index=ans_ids) # [b, 2]
            neg_metrics = (neg_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
            neg_acc[attn_sink_layer] += (neg_metrics > eps_threshold).sum().item()
        
    # ###### Report ######
    print("Final results:")
    print(f"Total valid samples: {full_num_entries}")
    acc_per_layer = [neg_acc[layer] / full_num_entries for layer in range(model.cfg.n_layers)]
    print(f"Accuracies by layer: [{', '.join(f'{acc:.4f}' for acc in acc_per_layer)}]")
    max_acc = max(acc_per_layer)
    max_layer = acc_per_layer.index(max_acc)
    print(f"Maximum accuracy: {max_acc:.4f} at layer {max_layer}")
    print(f"Neg vanilla accuracy: {neg_vanilla_acc / full_num_entries:.4f}")
    print()

    if args.should_save:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        # just pickle the neg acc as a list of floats
        output_file = output_dir / f"attn_sink_base_models_gen_neg_acc.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(neg_acc, f)
        print(f"\nSaved results to: {output_file}")
        print(f"Saved data includes:")
        print(f"  - neg_acc: shape {len(neg_acc)}")
