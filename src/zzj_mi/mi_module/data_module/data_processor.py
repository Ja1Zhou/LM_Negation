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

import re
import torch as t
class DataProcessor:
    logged_sanity_check = False
    NEG_INDICATOR_PATTERN = re.compile(r"\b(cannot|not|no)\b")

    def __init__(self):
        self.device = "cuda" if t.cuda.is_available() else \
            "mps" if t.backends.mps.is_available() else "cpu"

    @staticmethod
    def get_token_span(
        tok_span: list[tuple[int, int]],
        str_span: tuple[int, int],
    ):
        ss, se = str_span
        length = len(tok_span)
        si = None
        ei = None
        for i, (tok_s, tok_e) in enumerate(tok_span):
            if tok_e > tok_s and tok_s <= ss < tok_e:
                si = i
                break
        if si is None:
            raise ValueError(f"Could not find start token span for {str_span}")
        for i in range(si, length):
            tok_s, tok_e = tok_span[i]
            if tok_e > tok_s and tok_s <= se < tok_e:
                ei = i
                break
        if ei is None:
            raise ValueError(f"Could not find end token span for {str_span}")
        return list(range(si, ei + 1))

    def filter_neg_indicator(
        self,
        neg_q: str,
        offset_mapping: list[tuple[int, int]],
    ):
        matches = self.NEG_INDICATOR_PATTERN.findall(neg_q)
        assert len(matches) == 1, f"Expected 1 match, got {len(matches)} for {neg_q}\n"
        indicator = matches[0]
        si = neg_q.find(indicator)
        se = si + len(indicator) - 1
        return self.get_token_span(offset_mapping, (si, se))

    def refine_prompts_and_answer(
        self,
        pos_prompt_ids: list[int],
        neg_prompt_ids: list[int],
        pos_ans_ids: list[int],
        neg_ans_ids: list[int],
    ):
        # this function refines prompt ids and answer ids
        prefix, refined_pos_ans_ids, refined_neg_ans_ids = self.find_diverging_token(pos_ans_ids, neg_ans_ids)
        refined_pos_prompt_ids = pos_prompt_ids + prefix
        # refined_pos_prompt_attn_mask = [1] * len(refined_pos_prompt_ids)
        refined_neg_prompt_ids = neg_prompt_ids + prefix
        # refined_neg_prompt_attn_mask = [1] * len(refined_neg_prompt_ids)
        return (
            refined_pos_prompt_ids,
            refined_neg_prompt_ids,
            refined_pos_ans_ids,
            refined_neg_ans_ids,
        )

    @staticmethod
    def find_diverging_token(
        pos_ans_ids: list[int],
        neg_ans_ids: list[int],
    ):
        # return shared prefix and first diverging token for pos and neg answer ids
        to_return = list()
        for pid, nid in zip(pos_ans_ids, neg_ans_ids):
            if pid != nid:
                return to_return, pid, nid
            to_return.append(pid)
        # everything is the same
        raise ValueError("Everything is the same")
