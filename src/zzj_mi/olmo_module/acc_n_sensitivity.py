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
Calculate and save positive accuracy, negative accuracy, and sensitivity metric
for OLMo checkpoints at different training steps.

The sensitivity metric is defined as:
(neg_ans - pos_ans)_{neg_prompt} - (neg_ans - pos_ans)_{pos_prompt}

If this metric is positive, the model is sensitive to negation.
"""
# ###### Import packages ######
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
from jaxtyping import Float

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_gen import DataProcessor
from tqdm import tqdm

# ###### Configs ######
DEFAULT_FLOAT_DTYPE = t.bfloat16

if __name__ == "__main__":
    # we do not need gradients
    t.set_grad_enabled(False)
    # ###### Parse args ######
    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda" if t.cuda.is_available() else ("mps" if t.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

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
        revision=args.model_revision, # this is specific to olmo for now
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

    eps_threshold = t.tensor(1e-6, dtype=DEFAULT_FLOAT_DTYPE, device=device)

    # Track scores
    neg_acc = t.tensor(0, dtype=t.int64, device=device)
    pos_acc = t.tensor(0, dtype=t.int64, device=device)
    sensitivity_acc = t.tensor(0, dtype=t.int64, device=device)

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
        neg_acc += (neg_metrics > eps_threshold).sum(dtype=t.int64)

        pos_logits = model(
            input=pos_inputs['input_ids'],
            attention_mask=pos_inputs['attention_mask'],
        ) # pos_logits: [b, seq_len, vocab_size]
        pos_logits = pos_logits[:, -1, :]
        pos_logits = cast(Float[t.Tensor, "b vocab_size"], pos_logits)

        pos_logits_of_interest = pos_logits.gather(dim=-1, index=ans_ids) # [b, 2]
        pos_metrics = (pos_logits_of_interest * neg_minus_pos_logits).sum(dim=-1) # [b]
        pos_acc += (pos_metrics < -eps_threshold).sum(dtype=t.int64)

        logits_diff = neg_metrics - pos_metrics # [b]
        sensitivity_acc += (logits_diff > eps_threshold).sum(dtype=t.int64)

    # finally, we should normalize the metrics 
    neg_acc = (neg_acc / full_num_entries).item()
    pos_acc = (pos_acc / full_num_entries).item()
    sensitivity_acc = (sensitivity_acc / full_num_entries).item()

    # ###### Report ######
    print(f"Pos accuracy: {pos_acc:.4f}")
    print(f"Neg accuracy: {neg_acc:.4f}")
    print(f"Sensitivity accuracy: {sensitivity_acc:.4f}")
    print()

    # ###### Save results ######
    if args.should_save:
        output_dir = args.get_complete_output_path_with_revision()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"acc_n_sensitivity.pt"
        t.save({
            'neg_acc': neg_acc,
            'pos_acc': pos_acc,
            'sensitivity_acc': sensitivity_acc
        }, output_file)
        print(f"\nSaved results to: {output_file}")
        print(f"Saved data includes:")
        print(f"  - neg_acc: {neg_acc:.4f}")
        print(f"  - pos_acc: {pos_acc:.4f}")
        print(f"  - sensitivity_acc: {sensitivity_acc:.4f}")
        print()

