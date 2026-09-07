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
Windowed Attention Sink ablation on the multi-answer data
(paper Figure 4 and the "Attention Sink Ablation" panels of Figures 6 and 7).

Cache the attention patterns of the negative prompts at every layer. For each
window size W in 1..5 and each center layer, run the negative prompt with the
attention sink applied at the last position of layers L..L+W-1, while every
other layer's attention pattern is held fixed at its cached value. Records,
per configuration, the top-k tokens and the average logits of the negative
and positive answer sets at the last position; the last configuration is the
unablated control.

Writes <output_path>/<model>/mi/attn_sink_ablation_multil_sweepl_w{1..5}_fix_scores_sweep.pt
when --should_save true.

    python src/zzj_mi/mi_module/attn_sink_ablation_multil_sweepl_fix_scores_expanded.py \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \
        --output_path outputs_release/ --should_save true
"""
# ###### Import packages ######
import re
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

def last_pos_layer_attn_sink_patch(
    attn_scores: Float[t.Tensor, "b h q k"],
    hook: HookPoint,
    bos_pos: Int[t.Tensor, "b"],
):
    """
    Disable attention at last position except for:
    1) itself (last key position)
    2) first non-padding token (per-batch bos_pos)
    
    This effectively disables the attention mechanism while keeping minimal context.
    """
    b, h, q, k = attn_scores.shape

    # Clone and modify only the last query row
    last_q_scores = attn_scores[:, :, -1, :].clone()  # [b, h, k]

    # Fill with -inf, then restore specific positions
    last_q_scores.fill_(t.finfo(attn_scores.dtype).min)

    # Allow: 1) self (last key), 2) first non-padding token
    allowed_pos = t.stack((t.full_like(bos_pos, -1), bos_pos), dim=0)  # [2, b]
    last_q_scores[t.arange(b), :, allowed_pos] = attn_scores[t.arange(b), :, -1, allowed_pos]
    
    attn_scores[:, :, -1, :] = last_q_scores
    return attn_scores

def fix_attn_scores_with_multi_ablation(
    attn_scores: Float[t.Tensor, "b h q k"],
    hook: HookPoint,
    neg_cache: dict,
    ablated_layers: set[int],
    bos_pos: Int[t.Tensor, "b"],
):
    """
    Fix attention scores to cached values for all layers except the ablated layers.
    At ablated layers, apply attention sink (disable attention at last position).
    
    Args:
        attn_scores: Current attention scores [b, h, q, k]
        hook: Hook point
        neg_cache: Cache containing normal attention scores from all layers
        ablated_layers: Set of layer indices to ablate (disable attention)
        bos_pos: Index of first non-pad token for each batch item [b]
    
    Returns:
        Fixed or ablated attention scores
    """
    current_layer = get_layer_from_hook_name(hook.name)
    
    if current_layer in ablated_layers:
        # Apply attention sink at this layer (ablate attention at last position)
        return last_pos_layer_attn_sink_patch(attn_scores, hook, bos_pos)
    else:
        # Fix to cached values (prevent backup attention)
        cached_name = get_act_name("attn_scores", current_layer)
        return neg_cache[cached_name]

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
    print(f"Strategy: Fix all non-ablated layers' attention scores to prevent backup attention")
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
        
        # Calculate bos position for attention sink (first non-pad token)
        neg_bos_pos = t.argmax(neg_inputs['attention_mask'], dim=-1)  # [b]

        # Prepare answer IDs for metric calculation
        pos_ans_ids = pos_anss['input_ids']
        neg_ans_ids = neg_anss['input_ids']

        # ###### Step 1: Cache all attention scores from normal forward pass ######
        attn_score_names = [get_act_name("attn_scores", layer) for layer in range(model.cfg.n_layers)]
        
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
                num_sweep_points = config['num_sweep_points']
                config['to_save_top_tokens'] = t.empty(
                    num_sweep_points, full_num_entries, TOPK,
                    dtype=t.int64, device=device
                )
                config['to_save_avg_logits'] = t.empty(
                    num_sweep_points, full_num_entries, 3,
                    dtype=DEFAULT_FLOAT_DTYPE, device=device
                )

        # ###### Step 2: Outer sweep loop - iterate over window sizes ######
        for w_size in WINDOW_SIZES:
            config = window_size_configs[w_size]
            num_sweep_points = config['num_sweep_points']
            batch_top_tokens = t.empty(num_sweep_points, batch_size, TOPK, dtype=t.int64, device=device)
            batch_avg_logits = t.empty(num_sweep_points, batch_size, 3, dtype=DEFAULT_FLOAT_DTYPE, device=device)

            # Inner loop - ablate each window with current window size
            for sweep_idx in range(num_sweep_points):
            
                if sweep_idx < num_sweep_points - 1:
                    # Apply fixed attention scores with ablation at target window
                    ablated_layers = set(range(sweep_idx, sweep_idx + w_size))
                
                    patch_fn = partial(
                        fix_attn_scores_with_multi_ablation,
                        neg_cache=neg_cache,
                        ablated_layers=ablated_layers,
                        bos_pos=neg_bos_pos
                    )
                    
                    neg_logits = model.run_with_hooks(
                        input=neg_inputs['input_ids'],
                        attention_mask=neg_inputs['attention_mask'],
                        fwd_hooks=[(is_attn_scores_hook, patch_fn)],
                    )
                else:
                    # Control: no attention sink (normal forward pass)
                    neg_logits = model(
                        input=neg_inputs['input_ids'],
                        attention_mask=neg_inputs['attention_mask'],
                    )
                
                # Extract final position logits
                neg_final_logits = neg_logits[:, -1, :]  # [b, d_vocab]
                neg_final_logits = cast(Float[t.Tensor, "b d_vocab"], neg_final_logits)

                # Get top-k tokens at final layer
                top_k_tokens = neg_final_logits.topk(TOPK).indices  # [b, TOPK]
                batch_top_tokens[sweep_idx] = top_k_tokens

                # Calculate average logits for answers
                # Positive answers
                pos_ans_logits = neg_final_logits.gather(dim=-1, index=pos_ans_ids)  # [b, padded_len]
                pos_ans_logits = pos_ans_logits * pos_anss['attention_mask']
                pos_ans_sum = pos_ans_logits.sum(dim=-1)  # [b]
                pos_ans_avg = pos_ans_sum / pos_anss['attention_mask'].sum(dim=-1)  # [b]

                # Negative answers
                neg_ans_logits = neg_final_logits.gather(dim=-1, index=neg_ans_ids)  # [b, padded_len]
                neg_ans_logits = neg_ans_logits * neg_anss['attention_mask']
                neg_ans_sum = neg_ans_logits.sum(dim=-1)  # [b]
                neg_ans_avg = neg_ans_sum / neg_anss['attention_mask'].sum(dim=-1)  # [b]

                # All answers (average)
                all_ans_avg = (pos_ans_sum + neg_ans_sum) / (
                    pos_anss['attention_mask'].sum(dim=-1) + neg_anss['attention_mask'].sum(dim=-1)
                )  # [b]

                # Stack: [b, 3] where 3 = [all_ans, pos_ans, neg_ans]
                avg_logits = t.stack([all_ans_avg, pos_ans_avg, neg_ans_avg], dim=-1)  # [b, 3]
                batch_avg_logits[sweep_idx] = avg_logits

                # Track accuracy (neg_ans_avg > pos_ans_avg)
                correct = (neg_ans_avg - pos_ans_avg) > eps_threshold
                config['accuracy_tracker'][sweep_idx] += correct.sum().item()

            # Store batch results for this window size
            config['to_save_top_tokens'][:, batch_start_idx:batch_end_idx] = batch_top_tokens
            config['to_save_avg_logits'][:, batch_start_idx:batch_end_idx] = batch_avg_logits

        # Debug logging (first batch only, for window size 5)
        if not has_logged_results and 5 in WINDOW_SIZES:
            w_size = 5
            config = window_size_configs[w_size]
            num_sweep_points = config['num_sweep_points']
            batch_top_tokens = config['to_save_top_tokens'][:, batch_start_idx:batch_end_idx]
            batch_avg_logits = config['to_save_avg_logits'][:, batch_start_idx:batch_end_idx]
            
            has_logged_results = True
            first_neg_prompt = data['negative_q'][0]
            first_neg_answer = data['negative_a'][0]
            first_pos_answer = data['positive_a'][0]
            
            print(f"\n=== First example (Window Size {w_size}) ===")
            print(f"NEGATIVE prompt: {first_neg_prompt}")
            print(f"  Should predict: **{first_neg_answer}**")
            print(f"  Wrong answer: {first_pos_answer}")
            print()
            
            print(f"Top-5 predictions when ablating different {w_size}-layer windows (with fixed scores):")
            print(f"{'Ablated Layers':<20} {'Top-5 Tokens':<60} {'Avg Logits (all, pos, neg)':<30} {'Correct?'}")
            print("-" * 130)
            
            # Show key checkpoints
            checkpoint_positions = [0, (num_sweep_points - 1) // 4, (num_sweep_points - 1) // 2, 
                                   3 * (num_sweep_points - 1) // 4, num_sweep_points - 2, num_sweep_points - 1]
            
            for sweep_idx in checkpoint_positions:
                if sweep_idx >= num_sweep_points:
                    continue
                    
                tokens = batch_top_tokens[sweep_idx, 0, :5]
                tokens_str = tok.convert_ids_to_tokens(tokens.tolist())
                logits = batch_avg_logits[sweep_idx, 0]
                is_correct = logits[2] > logits[1]  # neg_ans > pos_ans
                
                if sweep_idx == num_sweep_points - 1:
                    layer_label = "Control (none)"
                else:
                    start_layer = sweep_idx
                    end_layer = sweep_idx + w_size - 1
                    layer_label = f"Layers {start_layer}-{end_layer}"
                
                correct_mark = "✓" if is_correct else "✗"
                print(f"{layer_label:<20} {str(tokens_str):<60} {logits[0]:.3f}, {logits[1]:.3f}, {logits[2]:.3f}    {correct_mark}")
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
        print(f"Accuracy by ablated layer window (with fixed attention scores):")
        print(f"{'Ablated Layers':<20} {'Accuracy':<10} {'Change from Control'}")
        print("-" * 55)
        
        control_acc = accuracy_per_config[-1]
        for window_idx, acc in enumerate(accuracy_per_config):
            if window_idx == num_sweep_points - 1:
                layer_label = "Control (none)"
            else:
                start_layer = window_idx
                end_layer = window_idx + w_size - 1
                layer_label = f"Layers {start_layer}-{end_layer}"
            
            change = acc - control_acc
            change_str = f"{change:+.4f}" if window_idx < num_sweep_points - 1 else "baseline"
            print(f"{layer_label:<20} {acc:.4f}     {change_str}")
        
        print()
        
        # Find most critical window (biggest drop in accuracy)
        if num_sweep_points > 1:
            accuracy_changes = [accuracy_per_config[i] - control_acc for i in range(num_sweep_points - 1)]
            most_critical_idx = np.argmin(accuracy_changes)
            most_critical_drop = accuracy_changes[most_critical_idx]
            
            start_layer = most_critical_idx
            end_layer = most_critical_idx + w_size - 1
            print(f"Most critical window: layers {start_layer}-{end_layer} (accuracy drop: {most_critical_drop:.4f})")
            print(f"  → Ablating this {w_size}-layer window with fixed scores causes largest accuracy drop")
            print(f"  → These consecutive layers are most necessary (isolated from backup effects)")
        print()

    if args.should_save:
        output_dir = args.get_complete_output_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save separate file for each window size
        for w_size in WINDOW_SIZES:
            config = window_size_configs[w_size]
            num_sweep_points = config['num_sweep_points']
            accuracy_per_config = [acc / full_num_entries for acc in config['accuracy_tracker']]
            
            output_file = output_dir / f"attn_sink_ablation_multil_sweepl_w{w_size}_fix_scores_sweep.pt"
            
            # Build ablated_layers metadata
            ablated_layers_list = [
                list(range(i, i + w_size)) 
                for i in range(num_sweep_points - 1)
            ] + ['control']
            
            output_obj = {
                'top_tokens': config['to_save_top_tokens'].detach().clone().cpu(),
                'avg_logits': config['to_save_avg_logits'].detach().clone().cpu(),
                'ablated_layers': ablated_layers_list,
                'accuracy_per_config': accuracy_per_config,
                'n_layers': model.cfg.n_layers,
                'window_size': w_size,
                'data_path': args.data_path,
                'output_subdir': args.output_subdir,
            }
            t.save(output_obj, output_file)
            
            print(f"\n✓ Saved window size {w_size} results to: {output_file}")
            print(f"  - top_tokens: shape {config['to_save_top_tokens'].shape}")
            print(f"    Dimension 0: starting layer (0 to {num_sweep_points - 2}) + control")
            print(f"    Dimension 1: data sample")
            print(f"    Dimension 2: top-k tokens")
            print(f"  - avg_logits: shape {config['to_save_avg_logits'].shape}")
            print(f"    Dimension 0: starting layer")
            print(f"    Dimension 1: data sample")
            print(f"    Dimension 2: [all_ans, pos_ans, neg_ans]")
            print(f"  - accuracy_per_config: {len(accuracy_per_config)} values")
            print(f"  - window_size: {w_size}")
        
        print(f"\nTo analyze: python src/zzj_mi/mi_module/analysis_module/inspect_attn_sink_ablation_multil_sweepl_expanded.py")
