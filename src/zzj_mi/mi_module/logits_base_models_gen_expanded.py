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
Multi-answer evaluation (paper Table 8, tab:multi_ans).

Same three metrics as `logits_base_models_gen.py` (negative accuracy, positive
accuracy, sensitivity), but each prompt has several gold answers: the logit of
an answer set is the average first-token logit over its unique answers.

    python src/zzj_mi/mi_module/logits_base_models_gen_expanded.py \
        --model_path meta-llama/Llama-3.1-8B \
        --data_path data/prompts-cleaned-multi-release.json \
        --batch_size 324 --should_save false
"""
import random
import torch as t
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
)
from typing import cast
from datasets import load_dataset
from transformer_lens import HookedTransformer
from jaxtyping import Float

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_expanded import DataProcessor
from tqdm import tqdm


DEFAULT_FLOAT_DTYPE = t.bfloat16


def average_answer_logits(
    logits: Float[t.Tensor, "b vocab_size"],
    anss: dict[str, t.Tensor],
) -> Float[t.Tensor, "b"]:
    ans_logits = logits.gather(dim=-1, index=anss["input_ids"])
    ans_logits = ans_logits * anss["attention_mask"]
    ans_sum = ans_logits.sum(dim=-1)
    ans_count = anss["attention_mask"].sum(dim=-1)
    return ans_sum / ans_count


if __name__ == "__main__":
    t.set_grad_enabled(False)

    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)

    print("Working with model:")
    print(args.model_path)
    print()

    random.seed(args.seed)
    t.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)

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
        trust_remote_code=True,
    )

    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True

    eps_threshold = 1e-6
    neg_acc = 0
    pos_acc = 0
    sensitivity_acc = 0

    total_batches = (full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0)
    for data in tqdm(full_data.iter(batch_size=args.batch_size), total=total_batches):
        pos_inputs, neg_inputs, pos_anss, neg_anss = data_processor.get_model_inputs(data, tok)

        neg_logits = model(input=neg_inputs["input_ids"], attention_mask=neg_inputs["attention_mask"])
        neg_logits = cast(Float[t.Tensor, "b vocab_size"], neg_logits[:, -1, :])
        pos_logits = model(input=pos_inputs["input_ids"], attention_mask=pos_inputs["attention_mask"])
        pos_logits = cast(Float[t.Tensor, "b vocab_size"], pos_logits[:, -1, :])

        # logit difference (negative answers minus positive answers) under each prompt
        neg_metric = average_answer_logits(neg_logits, neg_anss) - average_answer_logits(neg_logits, pos_anss)
        pos_metric = average_answer_logits(pos_logits, neg_anss) - average_answer_logits(pos_logits, pos_anss)

        neg_acc += (neg_metric > eps_threshold).sum().item()
        pos_acc += (pos_metric < -eps_threshold).sum().item()
        sensitivity_acc += ((neg_metric - pos_metric) > eps_threshold).sum().item()

    print("Final results:")
    print(f"Total valid samples: {full_num_entries}")
    print(f"Neg accuracy: {neg_acc / full_num_entries:.4f}")
    print(f"Pos accuracy: {pos_acc / full_num_entries:.4f}")
    print(f"Sensitivity accuracy: {sensitivity_acc / full_num_entries:.4f}")
