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
src/zzj_mi/mi_module/data_module/base_models_expanded.py
NOTE:
this is actually for base model gen with expanded data
"""
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import random
import torch as t

import zzj_mi.mi_module.data_module.data_processor as base_module

class DataProcessor(base_module.DataProcessor):
    """
    Data processor for base models using direct tokenization
    """
    def process_dataset(
        self,
        data: Dataset,
        tok: PreTrainedTokenizerFast,
    ):
        neg_prompts = list(data["negative_q"])
        neg_prompt_toked = tok(
            neg_prompts,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        neg_indicator_token_positions = [
            self.filter_neg_indicator(neg_q, offset_mapping)
            for neg_q, offset_mapping in zip(neg_prompts, neg_prompt_toked["offset_mapping"])
        ]
        num_neg_indicator_tokens = [len(pos) for pos in neg_indicator_token_positions]
        data = data.add_column("neg_indicator_token_positions", neg_indicator_token_positions)
        data = data.add_column("num_neg_indicator_tokens", num_neg_indicator_tokens)
        return data

    def remove_duplicate_toks(
        self,
        pos_ans_ids: list[int],
        neg_ans_ids: list[int],
    ):
        uniq_pos_as = set(pos_ans_ids)
        uniq_neg_as = set(neg_ans_ids)
        dup = uniq_pos_as.intersection(uniq_neg_as)
        uniq_pos_as = uniq_pos_as - dup
        uniq_neg_as = uniq_neg_as - dup
        if len(uniq_pos_as) == 0 or len(uniq_neg_as) == 0:
            print("No remaining unique tokens")
            print(f"Pos ans ids: {pos_ans_ids}")
            print(f"Neg ans ids: {neg_ans_ids}")
            raise ValueError("No remaining unique tokens")
        return list(uniq_pos_as), list(uniq_neg_as)
    
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
        # Expanded data: each prompt has a list of answers
        """
        Here data['positive_a'] is a list
        - for each data entry, create a list of prompt + answer
        - tokenize each list separately for clean logic
        """
        pos_prompt_pos_ans: list[list[str]] = [
            [f"{p} {a}" for a in pos_a] for p, pos_a in zip(pos_prompts, data['positive_a']) # every prompt gets a list of prompt + answer
        ]
        # for every prompt, we get a list of list of ints
        pos_prompt_pos_ans_ids: list[list[list[int]]] = [tok(p)['input_ids'] for p in pos_prompt_pos_ans]
        # extract pos ans ids
        # first we need to zip the first level entry with prompt length
        pos_ans_ids: list[list[int]] = [
            [p[len_p] for p in pid] for pid, len_p in zip(pos_prompt_pos_ans_ids, pos_prompt_lens) # to get the first token
        ] # for every prompt, we have a list of first token ids
        
        pos_prompt_neg_ans: list[list[str]] = [
            [f"{p} {a}" for a in neg_a] for p, neg_a in zip(pos_prompts, data['negative_a']) # every prompt gets a list of prompt + answer
        ]
        pos_prompt_neg_ans_ids: list[list[list[int]]] = [tok(p)['input_ids'] for p in pos_prompt_neg_ans]
        # extract pos ans ids
        # first we need to zip the first level entry with prompt length
        neg_ans_ids: list[list[int]] = [
            [p[len_p] for p in pid] for pid, len_p in zip(pos_prompt_neg_ans_ids, pos_prompt_lens) # to get the first token
        ]

        """
        Now we need to refine the pos and neg ans lists
        - For each prompt, we have recorded the first tokens
        - make them both sets, and remove the intersection
        """
        pos_ans_ids, neg_ans_ids = zip(*[self.remove_duplicate_toks(p, n) for p, n in zip(pos_ans_ids, neg_ans_ids)])
        
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

        pos_ans_ids = tok.pad(
            {"input_ids": pos_ans_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)

        neg_ans_ids = tok.pad(
            {"input_ids": neg_ans_ids},
            padding_side="left",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Debug logging (only once)
        if not self.logged_sanity_check:
            random_idx = random.randint(0, len(pos_prompt_ids)-1)
            peek_pos_prompt_ids = pos_inputs['input_ids'][random_idx].tolist()
            peek_pos_attn_mask = pos_inputs['attention_mask'][random_idx].tolist()
            peek_neg_prompt_ids = neg_inputs['input_ids'][random_idx].tolist()
            peek_neg_attn_mask = neg_inputs['attention_mask'][random_idx].tolist()
            peek_pos_ans_ids = pos_ans_ids['input_ids'][random_idx].tolist()
            peek_pos_ans_attn_mask = pos_ans_ids['attention_mask'][random_idx].tolist()
            peek_neg_ans_ids = neg_ans_ids['input_ids'][random_idx].tolist()
            peek_neg_ans_attn_mask = neg_ans_ids['attention_mask'][random_idx].tolist()
            
            print("Sanity check of get_model_inputs():")
            print(f"Pos: {pos_prompts[random_idx]} **{data['positive_a'][random_idx]}**")
            print(f"Neg: {neg_prompts[random_idx]} **{data['negative_a'][random_idx]}**")
            print(f"Pos prompt IDs: {tok.convert_ids_to_tokens(peek_pos_prompt_ids)}")
            print(f"Pos attn mask: {peek_pos_attn_mask}")
            print()
            print(f"Neg prompt IDs: {tok.convert_ids_to_tokens(peek_neg_prompt_ids)}")
            print(f"Neg attn mask: {peek_neg_attn_mask}")
            print()
            print(f"Pos ans ids: {tok.convert_ids_to_tokens(peek_pos_ans_ids)}")
            print(f"Pos ans attn mask: {peek_pos_ans_attn_mask}")
            print(f"Neg ans ids: {tok.convert_ids_to_tokens(peek_neg_ans_ids)}")
            print(f"Neg ans attn mask: {peek_neg_ans_attn_mask}")
            print()
            self.logged_sanity_check = True
            
        return pos_inputs, neg_inputs, pos_ans_ids, neg_ans_ids

if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer
    full_data = load_dataset("json", data_files="data/prompts-cleaned-multi-release.json", split="train")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Simple test
    data_processor = DataProcessor()
    print("Base model DataProcessor created successfully")

    for data in full_data.iter(batch_size=10):
        pos_inputs, neg_inputs, pos_ans_ids, neg_ans_ids = data_processor.get_model_inputs(data, tok)
