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
Windowed attention-output path patching on the multi-answer data
(paper Figures 6 and 7, "Attention Output Path Patching" panels).

Cache the attention outputs of the positive prompts and the attention
patterns of the negative prompts at every layer. For each window size W in
1..5 and each center layer, run the negative prompt with the attention
outputs of layers L..L+W-1 replaced by the positive prompt's, while every
layer's attention pattern is held fixed at its cached negative value (so no
other head can compensate); MLPs are recomputed. Records, per configuration,
the top-k tokens and the average logits of the negative and positive answer
sets at the last position, from which the viz derives the surrogate negation
accuracy. The last configuration is the unpatched control.

Writes <output_path>/<model>/mi/patch_attn_multil_sweepl_w{1..5}_fix_scores_sweep.pt
when --should_save true.

    python src/zzj_mi/mi_module/patch_attn_multil_sweepl_fix_scores_expanded.py \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \
        --output_path outputs_release/ --should_save true
"""
# ###### Import packages ######
import numpy as np
import torch as t
import re
from functools import partial
from pathlib import Path
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
from zzj_mi.mi_module.data_module.base_models_expanded import DataProcessor
from tqdm import tqdm

DEFAULT_FLOAT_DTYPE = t.bfloat16
TOPK = 10
WINDOW_SIZES = [1, 2, 3, 4, 5]  # Sweep window sizes from 1 to 5

# ###### Attention Score Functions ######
_HOOK_RE = re.compile(r"^blocks\.(\d+)\..*hook_attn_scores$")

def get_layer_from_hook_name(hook_name: str) -> int:
    """Extract layer index from hook name."""
    m = _HOOK_RE.match(hook_name)
    if m is None:
        raise ValueError(f"Cannot extract layer from hook name: {hook_name}")
    return int(m.group(1))

def is_attn_scores_hook(hook_name: str) -> bool:
    """Filter function to identify attention scores hooks."""
    return _HOOK_RE.match(hook_name) is not None

def fix_attn_scores_hook(
    attn_scores: Float[t.Tensor, "b h q k"],
    hook: HookPoint,
    neg_cache: dict,
):
    """
    Fix attention scores to cached negative values for all layers.
    This prevents backup attention mechanisms from activating when outputs are patched.
    
    Args:
        attn_scores: Current attention scores [b, h, q, k]
        hook: Hook point
        neg_cache: Cache containing normal attention scores from all layers
    
    Returns:
        Fixed attention scores from cache
    """
    current_layer = get_layer_from_hook_name(hook.name)
    cached_name = get_act_name("attn_scores", current_layer)
    return neg_cache[cached_name]

def patch_attn_output_window(
    neg_attn_out: Float[t.Tensor, "b pos d_model"],
    hook: HookPoint,
    pos_cache: dict,
    patched_layers: set[int],
):
    """
    Patch attention output at last position only, if this layer is in the patched window.
    Replace negative prompt's attention output with positive prompt's.
    
    Args:
        neg_attn_out: Negative prompt attention output [b, pos, d_model]
        hook: Hook point
        pos_cache: Cache containing positive attention outputs from all layers
        patched_layers: Set of layer indices to patch
    
    Returns:
        Modified attention output with last position replaced if in patched layers
    """
    # Extract current layer from hook name
    current_layer = int(hook.name.split('.')[1])
    
    if current_layer in patched_layers:
        # Patch this layer: replace with positive attention output
        cached_name = get_act_name("attn_out", current_layer)
        pos_attn_out = pos_cache[cached_name].squeeze(1)  # [b, 1, d_model] -> [b, d_model]
        neg_attn_out[:, -1, :] = pos_attn_out
    
    return neg_attn_out

if __name__ == "__main__":
    # we do not need gradients
    t.set_grad_enabled(False)
    # ###### Parse args ######
    parser = HfArgumentParser(Argument)
    args = parser.parse_args_into_dataclasses()[0]
    args = cast(Argument, args)
    device = "cuda" if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"

    # ###### Load data ######
    full_data = load_dataset("json", data_files=args.data_path, split="train")
    full_num_entries = full_data.num_rows
    print(f"Loaded {full_num_entries} examples")

    # load tokenizer
    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    # ###### Load model ######
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=DEFAULT_FLOAT_DTYPE,
        default_padding_side="left",
        trust_remote_code=True,
    )

    # Initialize data processor
    data_processor = DataProcessor()
    if args.disable_sanity_check:
        data_processor.logged_sanity_check = True

    # Initialize storage for each window size
    # For each window size W, we have (n_layers - W + 1) starting positions + 1 control
    window_size_configs = {}
    for w_size in WINDOW_SIZES:
        num_positions = model.cfg.n_layers - w_size + 1
        window_size_configs[w_size] = {
            'num_sweep_points': num_positions + 1,  # +1 for control
            'to_save_top_tokens': None,
            'to_save_avg_logits': None,
            'accuracy_tracker': [0 for _ in range(num_positions + 1)],
        }

    print(f"Model has {model.cfg.n_layers} layers")
    print(f"Will sweep window sizes: {WINDOW_SIZES}")
    for w_size in WINDOW_SIZES:
        n_configs = window_size_configs[w_size]['num_sweep_points']
        print(f"  Window size {w_size}: {n_configs - 1} positions + control = {n_configs} configs")
    print(f"Strategy: Patch attention outputs in windows while fixing attention scores")
    print()

    eps_threshold = 1e-6

    # Outer loop: batches
    has_logged_results = False
    for batch_idx, data in enumerate(tqdm(full_data.iter(batch_size=args.batch_size), 
                                          total=(full_num_entries // args.batch_size) + int(full_num_entries % args.batch_size != 0))):
        batch_size = len(data['positive_q'])
        batch_start_idx = batch_idx * args.batch_size
        batch_end_idx = batch_start_idx + batch_size

        # Get model inputs
        pos_inputs, neg_inputs, pos_anss, neg_anss = data_processor.get_model_inputs(data, tok)

        # Prepare answer IDs for metric calculation
        pos_ans_ids = pos_anss['input_ids']
        neg_ans_ids = neg_anss['input_ids']

        # ###### Step 1: Cache positive attention outputs and negative attention scores ######
        attn_out_names = [get_act_name("attn_out", layer) for layer in range(model.cfg.n_layers)]
        attn_score_names = [get_act_name("attn_scores", layer) for layer in range(model.cfg.n_layers)]
        
        # Cache positive attention outputs at last position
        _, pos_cache = model.run_with_cache(
            input=pos_inputs['input_ids'],
            attention_mask=pos_inputs['attention_mask'],
            pos_slice=-1,  # Only cache last position
            names_filter=attn_out_names,
            return_type=None,
        )
        
        # Cache negative attention scores (all positions needed for fixing)
        _, neg_cache = model.run_with_cache(
            input=neg_inputs['input_ids'],
            attention_mask=neg_inputs['attention_mask'],
            names_filter=attn_score_names,
            return_type=None,
        )

        # Initialize batch storage for each window size
        for w_size in WINDOW_SIZES:
            config = window_size_configs[w_size]
            if config['to_save_top_tokens'] is None:
                config['to_save_top_tokens'] = t.empty(
                    config['num_sweep_points'], full_num_entries, TOPK,
                    dtype=t.int64, device=device
                )
                config['to_save_avg_logits'] = t.empty(
                    config['num_sweep_points'], full_num_entries, 3,
                    dtype=DEFAULT_FLOAT_DTYPE, device=device
                )

        # ###### Step 2: Outer sweep loop - iterate over window sizes ######
        for w_size in WINDOW_SIZES:
            config = window_size_configs[w_size]
            num_sweep_points = config['num_sweep_points']
            batch_top_tokens = t.empty(num_sweep_points, batch_size, TOPK, dtype=t.int64, device=device)
            batch_avg_logits = t.empty(num_sweep_points, batch_size, 3, dtype=DEFAULT_FLOAT_DTYPE, device=device)

            # Inner loop - patch each window with current window size
            for sweep_idx in range(num_sweep_points):
                
                if sweep_idx < num_sweep_points - 1:
                    # Patching configuration: patch layers [start_layer, start_layer + w_size - 1]
                    start_layer = sweep_idx
                    end_layer = start_layer + w_size - 1
                    patched_layers = set(range(start_layer, end_layer + 1))
                    
                    # Create hooks:
                    # 1. Fix all attention scores to cached negative values
                    # 2. Patch attention outputs for layers in the window
                    attn_score_hook = partial(fix_attn_scores_hook, neg_cache=neg_cache)
                    attn_out_hook = partial(patch_attn_output_window, pos_cache=pos_cache, patched_layers=patched_layers)
                    
                    fwd_hooks = [
                        (is_attn_scores_hook, attn_score_hook),
                        (lambda name: any(get_act_name("attn_out", layer) in name for layer in range(model.cfg.n_layers)),
                         attn_out_hook)
                    ]
                    
                    neg_logits = model.run_with_hooks(
                        input=neg_inputs['input_ids'],
                        attention_mask=neg_inputs['attention_mask'],
                        fwd_hooks=fwd_hooks,
                        return_type="logits",
                    )
                else:
                    # Control: no patching, but still fix attention scores
                    attn_score_hook = partial(fix_attn_scores_hook, neg_cache=neg_cache)
                    fwd_hooks = [(is_attn_scores_hook, attn_score_hook)]
                    
                    neg_logits = model.run_with_hooks(
                        input=neg_inputs['input_ids'],
                        attention_mask=neg_inputs['attention_mask'],
                        fwd_hooks=fwd_hooks,
                        return_type="logits",
                    )
                
                # Extract final position logits
                neg_final_logits = neg_logits[:, -1, :]  # [b, d_vocab]
                neg_final_logits = cast(Float[t.Tensor, "b d_vocab"], neg_final_logits)

                # Get top-k tokens at final layer
                top_k_tokens = neg_final_logits.topk(TOPK).indices  # [b, TOPK]
                batch_top_tokens[sweep_idx] = top_k_tokens

                # ###### Step 3: Calculate average logits for answers ######
                # Positive answers
                pos_ans_logits = neg_final_logits.gather(dim=-1, index=pos_ans_ids)
                pos_ans_logits = pos_ans_logits * pos_anss['attention_mask']
                pos_ans_sum = pos_ans_logits.sum(dim=-1)
                pos_ans_avg = pos_ans_sum / pos_anss['attention_mask'].sum(dim=-1)
                
                # Negative answers
                neg_ans_logits = neg_final_logits.gather(dim=-1, index=neg_ans_ids)
                neg_ans_logits = neg_ans_logits * neg_anss['attention_mask']
                neg_ans_sum = neg_ans_logits.sum(dim=-1)
                neg_ans_avg = neg_ans_sum / neg_anss['attention_mask'].sum(dim=-1)
                
                # All answers (concatenate pos and neg)
                all_ans_avg = (pos_ans_sum + neg_ans_sum) / (
                    pos_anss['attention_mask'].sum(dim=-1) + neg_anss['attention_mask'].sum(dim=-1)
                )
                
                # Stack into [b, 3]
                avg_logits = t.stack([all_ans_avg, pos_ans_avg, neg_ans_avg], dim=-1)
                batch_avg_logits[sweep_idx] = avg_logits

                # ###### Step 4: Calculate accuracy ######
                # Model is correct if pos_ans_avg > neg_ans_avg
                correct = (pos_ans_avg > neg_ans_avg + eps_threshold).float()
                config['accuracy_tracker'][sweep_idx] += correct.sum().item()

            # Store batch results for this window size
            config['to_save_top_tokens'][:, batch_start_idx:batch_end_idx] = batch_top_tokens
            config['to_save_avg_logits'][:, batch_start_idx:batch_end_idx] = batch_avg_logits

        # Debug logging (first batch only, for window size 5)
        if not has_logged_results and 5 in WINDOW_SIZES:
            has_logged_results = True
            w_size = 5
            config = window_size_configs[w_size]
            
            print("\n=== Debug: First batch results (Window size 5) ===")
            print(f"Batch size: {batch_size}")
            print(f"Number of sweep points: {config['num_sweep_points']}")
            
            # Show first sample results for a few configurations
            sample_idx = 0
            print(f"\nSample {sample_idx}:")
            print(f"  Positive question: {data['positive_q'][sample_idx][:80]}...")
            print(f"  Negative question: {data['negative_q'][sample_idx][:80]}...")
            
            for sweep_idx in [0, config['num_sweep_points'] // 2, config['num_sweep_points'] - 1]:
                if sweep_idx < config['num_sweep_points'] - 1:
                    start_layer = sweep_idx
                    end_layer = start_layer + w_size - 1
                    config_label = f"Layers {start_layer}-{end_layer}"
                else:
                    config_label = "Control (no patch)"
                
                top_tokens = batch_top_tokens[sweep_idx, sample_idx]
                avg_logits = batch_avg_logits[sweep_idx, sample_idx]
                
                print(f"\n  {config_label}:")
                print(f"    Top-3 tokens: {tok.decode(top_tokens[:3].tolist())}")
                print(f"    Avg logits [all, pos, neg]: {avg_logits.tolist()}")
                print(f"    Pos > Neg: {avg_logits[1] > avg_logits[2]}")
            print()

    # Calculate final accuracy for each window size
    print("\n=== Final Results ===")
    print(f"Total samples processed: {full_num_entries}")
    print()
    
    for w_size in WINDOW_SIZES:
        config = window_size_configs[w_size]
        num_sweep_points = config['num_sweep_points']
        accuracy_per_config = [acc / full_num_entries for acc in config['accuracy_tracker']]
        
        print(f"\n--- Window Size: {w_size} ---")
        print(f"Accuracy by patched layer window (with fixed attention scores):")
        print(f"{'Patched Layers':<20} {'Accuracy':<10} {'Change from Control'}")
        print("-" * 55)
        
        control_acc = accuracy_per_config[-1]
        for window_idx, acc in enumerate(accuracy_per_config):
            if window_idx < num_sweep_points - 1:
                start_layer = window_idx
                end_layer = start_layer + w_size - 1
                config_label = f"Layers {start_layer}-{end_layer}"
            else:
                config_label = "Control (no patch)"
            
            change = acc - control_acc
            change_str = f"{change:+.4f}" if window_idx < num_sweep_points - 1 else "baseline"
            print(f"{config_label:<20} {acc:.4f}     {change_str}")
        
        print()
        
        # Find most critical window (biggest increase in accuracy after patching)
        if num_sweep_points > 1:
            accuracy_changes = [accuracy_per_config[i] - control_acc for i in range(num_sweep_points - 1)]
            most_critical_idx = np.argmax(accuracy_changes)
            most_critical_gain = accuracy_changes[most_critical_idx]
            
            start_layer = most_critical_idx
            end_layer = start_layer + w_size - 1
            
            print(f"Most critical window: Layers {start_layer}-{end_layer} (accuracy gain: {most_critical_gain:+.4f})")
            print(f"  → Patching this window causes largest accuracy increase")
            print(f"  → These layers' attention outputs have strongest causal effect on negation errors")
        print()

    if args.should_save:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save separate file for each window size
        for w_size in WINDOW_SIZES:
            config = window_size_configs[w_size]
            num_sweep_points = config['num_sweep_points']
            accuracy_per_config = [acc / full_num_entries for acc in config['accuracy_tracker']]
            
            output_file = output_dir / f"patch_attn_multil_sweepl_w{w_size}_fix_scores_sweep.pt"
            
            output_obj = {
                'top_tokens': config['to_save_top_tokens'].detach().clone().cpu(),
                'avg_logits': config['to_save_avg_logits'].detach().clone().cpu(),
                'window_size': w_size,
                'patched_windows': [f"{i}-{i+w_size-1}" for i in range(num_sweep_points - 1)] + ['control'],
                'accuracy_per_config': accuracy_per_config,
                'n_layers': model.cfg.n_layers,
                'data_path': args.data_path,
                'output_subdir': args.output_subdir,
            }
            t.save(output_obj, output_file)
            
            print(f"✓ Saved window size {w_size} results to: {output_file}")
            print(f"  - top_tokens: shape {config['to_save_top_tokens'].shape}")
            print(f"    Dimension 0: patched window (0 to {num_sweep_points - 2}) + control")
            print(f"    Dimension 1: data sample")
            print(f"    Dimension 2: top-k tokens")
            print(f"  - avg_logits: shape {config['to_save_avg_logits'].shape}")
            print(f"    Dimension 0: patched window")
            print(f"    Dimension 1: data sample")
            print(f"    Dimension 2: [all_ans, pos_ans, neg_ans]")
            print(f"  - accuracy_per_config: {len(accuracy_per_config)} values")
        
        print(f"\nTo analyze: python src/zzj_mi/mi_module/analysis_module/inspect_patch_attn_multil_sweepl_expanded.py")
