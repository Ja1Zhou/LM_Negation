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
Data processor for base models (non-instruction models).
This handles direct tokenization without chat templates.

For most cases, we would need to return:
- pos prompt ids
    * pos attn mask
- neg prompt ids
    * neg attn mask
- pos ans ids
- neg ans ids
"""
from transformers import PreTrainedTokenizerFast
import random
import torch as t

import zzj_mi.mi_module.data_module.data_processor as base_module

class DataProcessor(base_module.DataProcessor):
    """
    Data processor for base models using direct tokenization
    """
    
    def get_model_inputs(
        self,
        data: dict,
        tok: PreTrainedTokenizerFast,
    ):
        """
        Process data for base models using direct tokenization approach.
        
        Args:
            data: Dictionary containing 'positive_q', 'negative_q', 'positive_a', 'negative_a'
            tok: Tokenizer
            
        Returns:
            tuple: (pos_inputs, neg_inputs, ans_ids)
                - pos_inputs: tokenized positive prompts with attention masks
                - neg_inputs: tokenized negative prompts with attention masks  
                - ans_ids: tensor of shape [b, 2] with [pos_ans_id, neg_ans_id] for each batch item
        """
        # Get prompts
        pos_prompts = list(data['positive_q'])
        pos_prompt_toked = tok(pos_prompts)
        pos_prompt_ids = pos_prompt_toked['input_ids']
        
        neg_prompts = list(data['negative_q'])
        neg_prompt_toked = tok(neg_prompts)
        neg_prompt_ids = neg_prompt_toked['input_ids']
        
        # Calculate pos and neg answer token ids
        pos_prompt_lens = [len(p) for p in pos_prompt_ids]
        
        # Create prompt + answer combinations
        pos_prompt_pos_ans = [
            f"{p} {a}" for p, a in zip(pos_prompts, data['positive_a'])
        ]
        pos_prompt_pos_ans_ids = tok(pos_prompt_pos_ans)['input_ids']
        
        pos_prompt_neg_ans = [
            f"{p} {a}" for p, a in zip(pos_prompts, data['negative_a'])
        ]
        pos_prompt_neg_ans_ids = tok(pos_prompt_neg_ans)['input_ids']

        # Extract answer token ids
        pos_ans_ids = [pid[len_p:] for pid, len_p in zip(pos_prompt_pos_ans_ids, pos_prompt_lens)]
        neg_ans_ids = [pid[len_p:] for pid, len_p in zip(pos_prompt_neg_ans_ids, pos_prompt_lens)]
        
        # Refine prompts and answers to handle shared prefixes and find diverging tokens
        refined_input_outputs = [
            self.refine_prompts_and_answer(pid, nid, p_ans_ids, n_ans_ids)
            for pid, nid, p_ans_ids, n_ans_ids in zip(pos_prompt_ids, neg_prompt_ids, pos_ans_ids, neg_ans_ids)
        ]

        pos_prompt_ids, neg_prompt_ids, pos_ans_ids, neg_ans_ids = zip(*refined_input_outputs)
        
        # Tokenize and pad
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
        
        # Convert answer ids to tensors
        pos_ans_ids = t.tensor(pos_ans_ids, dtype=t.int64)
        neg_ans_ids = t.tensor(neg_ans_ids, dtype=t.int64)
        ans_ids = t.stack([pos_ans_ids, neg_ans_ids], dim=-1).to(self.device)  # [b, 2]
        
        # Debug logging (only once)
        if not self.logged_sanity_check:
            random_idx = random.randint(0, len(pos_prompt_ids)-1)
            peek_pos_prompt_ids = pos_inputs['input_ids'][random_idx].tolist()
            peek_pos_attn_mask = pos_inputs['attention_mask'][random_idx].tolist()
            peek_neg_prompt_ids = neg_inputs['input_ids'][random_idx].tolist()
            peek_neg_attn_mask = neg_inputs['attention_mask'][random_idx].tolist()
            peek_ans_ids = ans_ids[random_idx].tolist()
            
            print("Sanity check of get_model_inputs():")
            print(f"Pos: {pos_prompts[random_idx]} **{data['positive_a'][random_idx]}**")
            print(f"Neg: {neg_prompts[random_idx]} **{data['negative_a'][random_idx]}**")
            print(f"Pos prompt IDs: {tok.convert_ids_to_tokens(peek_pos_prompt_ids)}")
            print(f"Pos attn mask: {peek_pos_attn_mask}")
            print()
            print(f"Neg prompt IDs: {tok.convert_ids_to_tokens(peek_neg_prompt_ids)}")
            print(f"Neg attn mask: {peek_neg_attn_mask}")
            print()
            print(f"Ans IDs (pos, neg): {tok.convert_ids_to_tokens(peek_ans_ids)}")
            print()
            self.logged_sanity_check = True
            
        return pos_inputs, neg_inputs, ans_ids

if __name__ == "__main__":
    # Simple test
    data_processor = DataProcessor()
    print("Base model DataProcessor created successfully")