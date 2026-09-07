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
src/zzj_mi/mi_module/logits_base_models_gen.py
"""
# ###### Import packages ######
import random
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
from jaxtyping import Int, Float

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_gen import DataProcessor
from tqdm import tqdm

# ###### Configs ######
DEFAULT_FLOAT_DTYPE = t.bfloat16
RANDOM_TOKEN_PAIRS = 10

def compute_mean_margin(neg_logits: Float[t.Tensor, "b vocab_size"], pos_logits: Float[t.Tensor, "b vocab_size"], ans_ids: Int[t.Tensor, "N 2"]) -> Float[t.Tensor, ""]:
    # ans_ids: [N, 2] columns [posTok, negTok]
    w = t.tensor([-1, 1], device=neg_logits.device, dtype=neg_logits.dtype)
    neg_m = (neg_logits.gather(-1, ans_ids) * w).sum(-1)  # [N]
    pos_m = (pos_logits.gather(-1, ans_ids) * w).sum(-1)  # [N]
    return (neg_m - pos_m).mean()  # scalar

if __name__ == "__main__":
    # we do not need gradients
    t.set_grad_enabled(False)
    # ###### Parse args ######
    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda"
    print(f"Working with model:")
    print(args.model_path)
    print()

    # ###### Set seed ######
    random.seed(args.seed)
    t.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)

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
    # disable sanity check
    data_processor.logged_sanity_check = True
    eps_threshold = 1e-6

    # Track scores
    neg_acc = 0
    pos_acc = 0
    sensitivity_acc = 0

    # default plan is to use a batch size over here
    for batch_idx, data in enumerate(tqdm(full_data.iter(batch_size=args.batch_size), total=(full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0))):
        batch_size = len(data['positive_q'])
        # ###### Calculate input ids and ans ids using DataProcessor ######
        pos_inputs, neg_inputs, ans_ids = data_processor.get_model_inputs(data, tok) # NOTE: pos ans first and then neg ans

        # first of all forward normally to get the logits difference
        neg_logits = model(
            input=neg_inputs['input_ids'],
            attention_mask=neg_inputs['attention_mask'],
        ) # neg_logits: [b, seq_len, vocab_size]
        neg_logits = neg_logits[:, -1, :]
        neg_logits = cast(Float[t.Tensor, "b vocab_size"], neg_logits)

        neg_logits_of_interest = neg_logits.gather(dim=-1, index=ans_ids) # [b, 2]
        neg_metrics = (neg_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
        neg_acc += (neg_metrics > eps_threshold).sum().item()

        pos_logits = model(
            input=pos_inputs['input_ids'],
            attention_mask=pos_inputs['attention_mask'],
        ) # pos_logits: [b, seq_len, vocab_size]
        pos_logits = pos_logits[:, -1, :]
        pos_logits = cast(Float[t.Tensor, "b vocab_size"], pos_logits)

        pos_logits_of_interest = pos_logits.gather(dim=-1, index=ans_ids) # [b, 2]
        pos_metrics = (pos_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
        pos_acc += (pos_metrics < -eps_threshold).sum().item()

        logits_diff = neg_metrics - pos_metrics # [b]
        sensitivity_acc += (logits_diff > eps_threshold).sum().item()

    # ###### Report ######
    print("Final results:")
    print(f"Total valid samples: {full_num_entries}")
    print(f"Neg accuracy: {neg_acc / full_num_entries:.4f}")
    print(f"Pos accuracy: {pos_acc / full_num_entries:.4f}")
    print(f"Sensitivity accuracy: {sensitivity_acc / full_num_entries:.4f}")
    print()