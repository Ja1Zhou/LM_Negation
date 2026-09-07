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
src/zzj_mi/mi_module/data_module/multi_template_expanded.py

DataProcessor for the 4-template expanded data
(`data/prompts-cleaned-multi-release.json`).

On top of the base `base_models_expanded.DataProcessor`, this processor
also returns per-example **padded** target token indices for the "last
token of `not Y`" position on the negative prompt and the matching
predicate-final position on the positive prompt.

Returned 6-tuple from `get_model_inputs`:
    pos_inputs: BatchEncoding (input_ids, attention_mask), left-padded
    neg_inputs: BatchEncoding (input_ids, attention_mask), left-padded
    pos_ans_ids: BatchEncoding (input_ids, attention_mask), left-padded
    neg_ans_ids: BatchEncoding (input_ids, attention_mask), left-padded
    pos_target_idx: LongTensor [batch_size], padded indices into pos_inputs
    neg_target_idx: LongTensor [batch_size], padded indices into neg_inputs
"""

from __future__ import annotations

import random

import torch as t
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

import zzj_mi.mi_module.data_module.base_models_expanded as base_module
from zzj_mi.mi_module.data_module.multi_template_target import (
    infer_template_id,
    resolve_target_token_idx,
)


class DataProcessor(base_module.DataProcessor):
    """Multi-template variant. Computes target token indices on top of the
    base behavior."""

    def get_prompt_inputs(
        self,
        data: dict,
        tok: PreTrainedTokenizerFast,
    ):
        """Tokenize + left-pad pos/neg prompts only. No answer-id work, no
        target indices. Used by the per-template save pipeline, which
        applies a fixed `pos_slice` per template instead of per-row gathers.
        """
        pos_prompt_ids = tok(list(data["positive_q"]))["input_ids"]
        neg_prompt_ids = tok(list(data["negative_q"]))["input_ids"]
        pos_inputs = tok.pad(
            {"input_ids": pos_prompt_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        neg_inputs = tok.pad(
            {"input_ids": neg_prompt_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        return pos_inputs, neg_inputs

    def _resolve_template_ids(self, data: dict) -> list[str]:
        if "template_id" in data:
            return list(data["template_id"])
        if "prompt_number" in data:
            return [infer_template_id(int(p)) for p in data["prompt_number"]]
        raise ValueError(
            "Multi-template DataProcessor requires either a 'template_id' field "
            "or a 'prompt_number' field on every row."
        )

    def get_model_inputs(
        self,
        data: dict,
        tok: PreTrainedTokenizerFast,
    ):
        if not tok.is_fast:
            raise RuntimeError(
                "Multi-template DataProcessor requires a fast tokenizer (offsets needed)."
            )
        pos_prompts = list(data["positive_q"])
        neg_prompts = list(data["negative_q"])
        template_ids = self._resolve_template_ids(data)

        pos_toked = tok(pos_prompts, return_offsets_mapping=True)
        neg_toked = tok(neg_prompts, return_offsets_mapping=True)
        pos_prompt_ids = pos_toked["input_ids"]
        neg_prompt_ids = neg_toked["input_ids"]

        # ----- target indices on unpadded sequences -----
        pos_targets_unpadded = [
            resolve_target_token_idx(p, tid, om)
            for p, tid, om in zip(pos_prompts, template_ids, pos_toked["offset_mapping"])
        ]
        neg_targets_unpadded = [
            resolve_target_token_idx(n, tid, om)
            for n, tid, om in zip(neg_prompts, template_ids, neg_toked["offset_mapping"])
        ]

        # ----- answer ids: same logic as base_models_expanded -----
        pos_prompt_lens = [len(p) for p in pos_prompt_ids]

        pos_prompt_pos_ans = [
            [f"{p} {a}" for a in pos_a]
            for p, pos_a in zip(pos_prompts, data["positive_a"])
        ]
        pos_prompt_pos_ans_ids = [tok(p)["input_ids"] for p in pos_prompt_pos_ans]
        pos_ans_ids: list[list[int]] = [
            [p[len_p] for p in pid]
            for pid, len_p in zip(pos_prompt_pos_ans_ids, pos_prompt_lens)
        ]

        pos_prompt_neg_ans = [
            [f"{p} {a}" for a in neg_a]
            for p, neg_a in zip(pos_prompts, data["negative_a"])
        ]
        pos_prompt_neg_ans_ids = [tok(p)["input_ids"] for p in pos_prompt_neg_ans]
        neg_ans_ids: list[list[int]] = [
            [p[len_p] for p in pid]
            for pid, len_p in zip(pos_prompt_neg_ans_ids, pos_prompt_lens)
        ]

        pos_ans_ids, neg_ans_ids = zip(
            *[self.remove_duplicate_toks(p, n) for p, n in zip(pos_ans_ids, neg_ans_ids)]
        )

        # ----- pad prompts -----
        pos_inputs = tok.pad(
            {"input_ids": pos_prompt_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        neg_inputs = tok.pad(
            {"input_ids": neg_prompt_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        # ----- pad answer id lists (so per-row lists stack into a tensor) -----
        pos_ans_ids_padded = tok.pad(
            {"input_ids": pos_ans_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        neg_ans_ids_padded = tok.pad(
            {"input_ids": neg_ans_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        # ----- shift unpadded -> padded target indices (left padding) -----
        pos_seq_len = pos_inputs["input_ids"].shape[1]
        neg_seq_len = neg_inputs["input_ids"].shape[1]
        pos_unpadded_lens = [len(ids) for ids in pos_prompt_ids]
        neg_unpadded_lens = [len(ids) for ids in neg_prompt_ids]
        pos_target_idx = t.tensor(
            [pos_seq_len - n + ti for n, ti in zip(pos_unpadded_lens, pos_targets_unpadded)],
            dtype=t.long,
            device=self.device,
        )
        neg_target_idx = t.tensor(
            [neg_seq_len - n + ti for n, ti in zip(neg_unpadded_lens, neg_targets_unpadded)],
            dtype=t.long,
            device=self.device,
        )

        # Sanity: target indices must fall on real (non-pad) tokens.
        pos_attn = pos_inputs["attention_mask"]
        neg_attn = neg_inputs["attention_mask"]
        batch_idx = t.arange(pos_target_idx.shape[0], device=self.device)
        if not bool((pos_attn[batch_idx, pos_target_idx] == 1).all()):
            raise RuntimeError("pos_target_idx points to a padding token")
        if not bool((neg_attn[batch_idx, neg_target_idx] == 1).all()):
            raise RuntimeError("neg_target_idx points to a padding token")

        if not self.logged_sanity_check:
            random_idx = random.randint(0, len(pos_prompt_ids) - 1)
            peek_pos_prompt_ids = pos_inputs["input_ids"][random_idx].tolist()
            peek_neg_prompt_ids = neg_inputs["input_ids"][random_idx].tolist()
            pos_pad_idx = int(pos_target_idx[random_idx].item())
            neg_pad_idx = int(neg_target_idx[random_idx].item())

            print("Sanity check of multi_template_expanded.DataProcessor.get_model_inputs():")
            print(f"  Template id: {template_ids[random_idx]}")
            print(f"  Pos prompt: {pos_prompts[random_idx]} **{data['positive_a'][random_idx]}**")
            print(f"  Neg prompt: {neg_prompts[random_idx]} **{data['negative_a'][random_idx]}**")
            print(f"  Pos prompt tokens: {tok.convert_ids_to_tokens(peek_pos_prompt_ids)}")
            print(f"  Neg prompt tokens: {tok.convert_ids_to_tokens(peek_neg_prompt_ids)}")
            print(f"  Pos target idx (padded): {pos_pad_idx} -> token "
                  f"{tok.convert_ids_to_tokens([peek_pos_prompt_ids[pos_pad_idx]])[0]!r}")
            print(f"  Neg target idx (padded): {neg_pad_idx} -> token "
                  f"{tok.convert_ids_to_tokens([peek_neg_prompt_ids[neg_pad_idx]])[0]!r}")
            print(f"  Pos target idx (unpadded): {pos_targets_unpadded[random_idx]} of "
                  f"{pos_unpadded_lens[random_idx]} (neg-index "
                  f"{pos_targets_unpadded[random_idx] - pos_unpadded_lens[random_idx]})")
            print(f"  Neg target idx (unpadded): {neg_targets_unpadded[random_idx]} of "
                  f"{neg_unpadded_lens[random_idx]} (neg-index "
                  f"{neg_targets_unpadded[random_idx] - neg_unpadded_lens[random_idx]})")
            print()
            self.logged_sanity_check = True

        return (
            pos_inputs,
            neg_inputs,
            pos_ans_ids_padded,
            neg_ans_ids_padded,
            pos_target_idx,
            neg_target_idx,
        )


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer

    full_data = load_dataset(
        "json",
        data_files="data/prompts-cleaned-multi-release.json",
        split="train",
    )
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dp = DataProcessor()
    print("Multi-template DataProcessor created.")

    for batch in full_data.iter(batch_size=8):
        outs = dp.get_model_inputs(batch, tok)
        print("Returned tuple length:", len(outs))
        print("pos_target_idx:", outs[4].tolist())
        print("neg_target_idx:", outs[5].tolist())
        break
