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
LogitLens sweep on negative prompts (paper Table 3 "LogitLens" column, Table 9).

For each layer i, the residual stream after layer i is passed through the
final layer norm and unembedding (skipping all later layers) and the negative
accuracy is recorded; the vanilla (full-model) accuracy is printed for reference.

    python src/zzj_mi/mi_module/logitlens_base_gen.py \
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

def filter_to_cache(act_name:str):
    return act_name.endswith("resid_pre") or act_name == "ln_final.hook_scale"

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
        neg_logits, neg_cache = model.run_with_cache(
            names_filter=filter_to_cache,
            input=neg_inputs['input_ids'],
            attention_mask=neg_inputs['attention_mask'],
            pos_slice=-1,
        ) # neg_logits: [b, seq_len, vocab_size]
        # gather logits for the last position; in order to do this we need to build neg and pos ans tokens
        neg_logits = cast(Float[t.Tensor, "b seq_len vocab_size"], neg_logits)
        neg_logits_of_interest = neg_logits[:, -1, :].gather(dim=-1, index=ans_ids) # [b, 2]
        neg_metrics = (neg_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
        neg_vanilla_acc += (neg_metrics > eps_threshold).sum().item()
        
        # ###### Calculate metrics ######
        for logitlens_layer in range(model.cfg.n_layers):
            layer_input = neg_cache[get_act_name("resid_pre", logitlens_layer)] # [b, 1, dmodel]
            # forward through the last ln
            # one option is to use cached scale
            ln_scale = neg_cache["ln_final.hook_scale"] # [b, 1, 1] -> [b, 1, dmodel]
            layer_logits = ((layer_input.to(t.float32) / ln_scale).to(DEFAULT_FLOAT_DTYPE) * model.ln_final.w) @ model.W_U + model.b_U
            # layer_input = model.ln_final(layer_input)
            # layer_logits = model.unembed(layer_input) # [b, 1, vocab_size]
            # gather logits for the last position; in order to do this we need to build neg and pos ans tokens
            neg_logits = cast(Float[t.Tensor, "b seq_len vocab_size"], layer_logits)
            neg_logits_of_interest = neg_logits[:, -1, :].gather(dim=-1, index=ans_ids) # [b, 2]
            neg_metrics = (neg_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
            neg_acc[logitlens_layer] += (neg_metrics > eps_threshold).sum().item()
        
    # ###### Report ######
    print("Final results:")
    print(f"Total valid samples: {full_num_entries}")
    acc_per_layer = [neg_acc[layer] / full_num_entries for layer in range(model.cfg.n_layers)]
    # for attn_sink_layer, acc in enumerate(acc_per_layer):
    #     print(f"Layer {attn_sink_layer}: {acc:.4f}")
    max_acc = max(acc_per_layer)
    max_layer = acc_per_layer.index(max_acc)
    print(f"Maximum accuracy: {max_acc:.4f} at layer {max_layer}")
    print(f"Neg vanilla accuracy: {neg_vanilla_acc / full_num_entries:.4f}")
    print()