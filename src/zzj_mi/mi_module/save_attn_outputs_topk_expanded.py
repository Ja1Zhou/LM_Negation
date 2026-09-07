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
Cache the most promoted (top-k) tokens of the attention outputs at layers 10-18 for every
negative prompt (paper Sec. 5.3-5.4; input to the LLM annotation behind
Figures 5 and 9).

For each layer in LAYER_RANGE the attention output at the last position is
projected onto the vocabulary with LogitLens and the topk token ids are kept,
giving a [num_entries, 9, TOPK] tensor plus the layer labels, saved to
<output_path>/<model>/mi/save_attn_outputs_topk_expanded.pt.

    python src/zzj_mi/mi_module/save_attn_outputs_topk_expanded.py \\
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \\
        --output_path outputs_release/ --should_save true
"""
# ###### Import packages ######
from pathlib import Path
import json
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
from jaxtyping import Float

from zzj_mi.arg_module.mi import Argument
from zzj_mi.mi_module.data_module.base_models_expanded import DataProcessor
from tqdm import tqdm

CUR_FILE_STEM = Path(__file__).stem
DEFAULT_FLOAT_DTYPE = t.bfloat16
TOPK = 10
LAYER_RANGE = list(range(10, 19)) # we save results from layer 10 to 18
LAYERS_TO_SAVE = [
    get_act_name("attn_out", layer) for layer in LAYER_RANGE
]
# the components are embed, attn, mlp, attn;
# 2 * layer + 1
COMP_IDX = [
    2 * layer + 1 for layer in LAYER_RANGE
]

if __name__ == "__main__":
    # we do not need gradients
    t.set_grad_enabled(False)
    # ###### Parse args ######
    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"

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
        trust_remote_code=True,
        device=device,
    )

    # Initialize data processor
    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True
    eps_threshold = 1e-6

    # Track results to save
    to_save = t.empty(full_num_entries, len(LAYERS_TO_SAVE), TOPK, dtype=t.int64, device=device) # [num_entries, num_layers, TOPK]

    # default plan is to use a batch size over here
    has_logged_results = False
    for batch_idx, data in enumerate(tqdm(full_data.iter(batch_size=args.batch_size), total=(full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0))):
        batch_size = len(data['positive_q'])
        batch_start_idx = batch_idx * args.batch_size
        batch_end_idx = batch_start_idx + batch_size

        # ###### Calculate input ids and ans ids using DataProcessor ######
        pos_inputs, neg_inputs, pos_anss, neg_anss = data_processor.get_model_inputs(data, tok)
        """
        NOTE: pos ans ids and neg ans ids are tensors padded
        pos ans ids is of shape [b, padded_len]
        """

        # Process negative inputs
        _, neg_cache = model.run_with_cache(
            input=neg_inputs['input_ids'],
            attention_mask=neg_inputs['attention_mask'],
            pos_slice=-1,
            return_type=None,
            names_filter=lambda x: x.endswith("hook_attn_out") or x.endswith("hook_mlp_out") or x == "ln_final.hook_scale" or x == "hook_embed",
        )

        neg_per_comp_resid, neg_per_comp_labels = neg_cache.decompose_resid(apply_ln=False, return_labels=True) # [num_comp, b, 1, dmodel]
        # take only the specified layers
        neg_per_comp_resid = neg_per_comp_resid[COMP_IDX]
        neg_final_ln_scale = neg_cache["ln_final.hook_scale"] # [b, 1, 1]
        neg_per_comp_resid_lned = (neg_per_comp_resid.to(t.float32) / neg_final_ln_scale).to(model.cfg.dtype) * model.ln_final.w # [num_comp, b, 1, dmodel]

        # mul with unembedding matrix for negative
        # model.W_U: [d_model, vocab_size] -> [num_comp, b, 1, dvocab]
        neg_per_comp_logits = neg_per_comp_resid_lned @ model.W_U # [num_comp, b, 1, dvocab]
        neg_per_comp_top_promote = neg_per_comp_logits.topk(TOPK).indices # [num_comp, b, 1, TOPK]
        # neg_per_comp_top_demote = neg_per_comp_logits.topk(TOPK, largest=False).indices
        # neg_accum_top_promote = neg_per_comp_logits.cumsum(dim=0).topk(TOPK).indices # [num_comp, b, 1, TOPK]
        # we have the topk indices to save
        to_save[batch_start_idx:batch_end_idx] = neg_per_comp_top_promote.squeeze(-2).swapaxes(0, 1).contiguous()

        if not has_logged_results:
            has_logged_results = True

            print("\n=== Debug: first negative example ===")
            print(f"NEG prompt: {data['negative_q'][0]}")
            print(f"NEG answer: {data['negative_a'][0]}\n")

            # pick first batch element
            example_topk = to_save[0]

            for layer_idx, tokens in zip(LAYER_RANGE, example_topk):
                tok_str = [tok.decode(t_id) for t_id in tokens.tolist()]
                print(f"Layer {layer_idx:02d}: {tok_str}")
            print()

        
    # ###### Report ######
    print("Final results:")
    print(f"Total valid samples: {full_num_entries}")
    print()

    if args.should_save:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{CUR_FILE_STEM}.pt"
        output_obj = {
            "attn_outputs_topk": to_save.detach().clone().cpu(),
            "labels": [neg_per_comp_labels[comp_idx] for comp_idx in COMP_IDX],
        }
        t.save(output_obj, output_file)
        print(f"\nSaved results to: {output_file}")
        print(f"Saved data includes:")
        print(f"  - attn_outputs_topk: shape {output_obj['attn_outputs_topk'].shape}")
        print(f"  - labels: {output_obj['labels']}")
        print()