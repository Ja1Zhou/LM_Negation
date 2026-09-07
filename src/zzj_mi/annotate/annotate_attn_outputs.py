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
LLM-as-a-judge annotation of attention outputs (paper Sec. 5.3-5.4, Figures 5 and 9).

Inputs (produced by `save_attn_outputs_topk_expanded.py` / `..._leastk_expanded.py`):
    <output_path>/<model>/mi/save_attn_outputs_{topk|leastk}_expanded.pt
Outputs:
    <output_path>/<model>/mi/annotate_attn_output{,_suppression}_expanded/sample_NNNN.json
    ... /summary.json   (per-layer evidence counts, layers 10-18)

Examples:
    # print the prompt for sample 0 and exit (no API call)
    python src/zzj_mi/annotate/annotate_attn_outputs.py --mode promote \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \
        --output_path outputs_release/ --dry_run
    # 3-sample smoke into a scratch directory
    python src/zzj_mi/annotate/annotate_attn_outputs.py --mode promote \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \
        --output_path outputs_release/ --output_subdir smoke --limit 3
    # full run (648 samples, resumable
    python src/zzj_mi/annotate/annotate_attn_outputs.py --mode promote \
        --model_path meta-llama/Llama-3.1-8B --data_path data/prompts-cleaned-multi-release.json \
        --output_path outputs_release/
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch as t
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizerFast

from zzj_mi.annotate.openrouter import DEFAULT_KEY_ENV, DEFAULT_MODEL, OpenRouterClient
from zzj_mi.arg_module.mi import Argument

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"
LAYER_RANGE = list(range(10, 19))  # must match save_attn_outputs_*_expanded.py
TOPK = 10

MODES = {
    "promote": dict(
        saved_file="save_attn_outputs_topk_expanded.pt",
        saved_key="attn_outputs_topk",
        template=PROMPTS_DIR / "annotate.md",
        out_dir="annotate_attn_output_expanded",
    ),
    "suppress": dict(
        saved_file="save_attn_outputs_leastk_expanded.pt",
        saved_key="attn_outputs_leastk",
        template=PROMPTS_DIR / "annotate_suppression.md",
        out_dir="annotate_attn_output_suppression_expanded",
    ),
}


@dataclass
class AnnotateArgument:
    mode: str = field(default="promote", metadata={"help": "promote (not-Y evidence in top-k) or suppress (Y evidence in least-k)"})
    api_model: str = field(default=DEFAULT_MODEL)
    api_key_env: str = field(default=DEFAULT_KEY_ENV)
    limit: int = field(default=0, metadata={"help": "annotate only the first N samples (0 = all)"})
    dry_run: bool = field(default=False, metadata={"help": "print the prompt for the first sample and exit"})


def format_annotation_prompt(neg_prompt: str, layer_tokens: dict[int, list[str]], instruction_template: str) -> str:
    attn_outputs_formatted = ""
    for layer_idx in LAYER_RANGE:
        attn_outputs_formatted += f"[{layer_idx}_attn_out]\n{layer_tokens.get(layer_idx, [])}\n\n"
    actual_input = f"""#### Actual Input

Prompt:
- {neg_prompt}

Attention outputs:
{attn_outputs_formatted}"""
    return instruction_template + "\n\n" + actual_input


def parse_json_reply(text: str):
    if "```json" in text:
        start = text.find("```json") + 7
        return json.loads(text[start:text.find("```", start)].strip())
    if "```" in text:
        start = text.find("```") + 3
        return json.loads(text[start:text.find("```", start)].strip())
    return json.loads(text)


def annotate_sample(client: OpenRouterClient, sample_idx: int, entry: dict, tokens_tensor: t.Tensor,
                    tokenizer: PreTrainedTokenizerFast, instruction_template: str, output_dir: Path) -> dict:
    result_file = output_dir / f"sample_{sample_idx:04d}.json"
    if result_file.exists():  # resume
        try:
            with open(result_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not load {result_file}: {e}; re-annotating")

    result = {
        "sample_idx": sample_idx,
        "prompt_number": entry.get("prompt_number"),
        "negative_q": entry.get("negative_q"),
        "negative_a": entry.get("negative_a"),
        "attr_1": entry.get("attr_1"),
        "attr_2": entry.get("attr_2"),
        "annotation_result": None,
        "annotation_raw": None,
        "error": None,
        "api_meta": None,
    }
    try:
        layer_tokens = {
            layer_idx: [tokenizer.decode(tid.item()) for tid in ids]
            for layer_idx, ids in zip(LAYER_RANGE, tokens_tensor)
        }
        prompt = format_annotation_prompt(entry["negative_q"], layer_tokens, instruction_template)
        raw, meta = client.chat([{"role": "user", "content": prompt}])
        result["annotation_raw"] = raw
        result["api_meta"] = meta.as_dict()
        try:
            result["annotation_result"] = parse_json_reply(raw)
        except json.JSONDecodeError as e:
            result["error"] = f"JSON parsing failed: {e}"
            print(f"Warning: failed to parse JSON for sample {sample_idx}: {e}")
    except Exception as e:
        result["error"] = f"API call failed: {e}"
        print(f"Error annotating sample {sample_idx}: {e}")

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def generate_summary(output_dir: Path, total_samples: int) -> dict:
    all_results = []
    for result_file in sorted(output_dir.glob("sample_*.json")):
        with open(result_file) as f:
            all_results.append(json.load(f))
    total_processed = len(all_results)
    successful = sum(1 for r in all_results if r.get("annotation_result") is not None)
    errors = sum(1 for r in all_results if r.get("error") is not None)
    samples_with_evidence = 0
    total_evidence_count = 0
    layer_evidence_counts = {layer: 0 for layer in LAYER_RANGE}
    for r in all_results:
        annotation = r.get("annotation_result")
        if annotation and isinstance(annotation, list) and len(annotation) > 0:
            samples_with_evidence += 1
            total_evidence_count += len(annotation)
            for evidence in annotation:
                if isinstance(evidence, dict) and evidence.get("layer") in layer_evidence_counts:
                    layer_evidence_counts[evidence["layer"]] += 1
    return {
        "total_samples": total_samples,
        "total_processed": total_processed,
        "successful_annotations": successful,
        "errors": errors,
        "samples_with_evidence": samples_with_evidence,
        "samples_without_evidence": successful - samples_with_evidence,
        "total_evidence_count": total_evidence_count,
        "avg_evidence_per_sample": total_evidence_count / total_processed if total_processed else 0,
        "layer_evidence_counts": layer_evidence_counts,
        "percentage_with_evidence": (samples_with_evidence / successful * 100) if successful else 0,
    }


def main() -> None:
    t.set_grad_enabled(False)
    parser = HfArgumentParser((Argument, AnnotateArgument))
    args, aargs = parser.parse_args_into_dataclasses()
    args = cast(Argument, args)
    aargs = cast(AnnotateArgument, aargs)
    if aargs.mode not in MODES:
        sys.exit(f"--mode must be one of {sorted(MODES)}")
    spec = MODES[aargs.mode]

    instruction_template = spec["template"].read_text()

    saved_results_file = args.get_complete_output_path() / spec["saved_file"]
    if not saved_results_file.exists():
        sys.exit(f"Missing {saved_results_file}; run the matching save_attn_outputs_*_expanded.py first.")
    saved = t.load(saved_results_file)
    tokens_all = saved[spec["saved_key"]]  # [num_entries, num_layers, TOPK]
    print(f"Loaded {spec['saved_key']}: {tuple(tokens_all.shape)} (labels: {saved.get('labels')})")

    full_data = load_dataset("json", data_files=args.data_path, split="train")
    n = full_data.num_rows
    if n != tokens_all.shape[0]:
        sys.exit(f"Data size mismatch: dataset has {n} rows, saved tensor has {tokens_all.shape[0]}.")

    tok: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    if aargs.dry_run:
        layer_tokens = {l: [tok.decode(i.item()) for i in ids] for l, ids in zip(LAYER_RANGE, tokens_all[0])}
        print(format_annotation_prompt(full_data[0]["negative_q"], layer_tokens, instruction_template))
        return

    client = OpenRouterClient(model=aargs.api_model, api_key_env=aargs.api_key_env)
    print(f"API: {client.describe()}")

    output_dir = args.get_complete_output_path() / spec["out_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    n_run = min(n, aargs.limit) if aargs.limit else n
    results = [
        annotate_sample(client, idx, full_data[idx], tokens_all[idx], tok, instruction_template, output_dir)
        for idx in tqdm(range(n_run), desc=f"Annotating ({aargs.mode})")
    ]
    tokens_in = sum((r.get("api_meta") or {}).get("prompt_tokens") or 0 for r in results)
    tokens_out = sum((r.get("api_meta") or {}).get("completion_tokens") or 0 for r in results)

    summary = generate_summary(output_dir, n)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60)
    print(f"{aargs.mode}: {summary['successful_annotations']}/{summary['total_processed']} annotated, "
          f"{summary['errors']} errors, {summary['percentage_with_evidence']:.1f}% with evidence")
    print("Evidence by layer: " + ", ".join(f"{l}:{summary['layer_evidence_counts'][l]}" for l in LAYER_RANGE))
    print(f"Tokens this run (new samples only): {tokens_in} in / {tokens_out} out")
    print(f"Summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
