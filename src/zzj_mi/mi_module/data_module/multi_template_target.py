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
Resolve the "last token of not Y" position (or its mirror on positive
prompts) for the 4 templates of data/prompts-cleaned-multi-release.json.

Each template has a fixed grammatical tail after the negated noun phrase:

  stem:           "Here is a list of X that are not Y:"          -> tail = ":"
  something_that: "Something that is an X and not a Y is"        -> tail = " is"
  article:        "An X that is not a Y is"                       -> tail = " is"
  question:       "What is an X that is not a Y? It is"           -> tail = "? It is"

The tail is stripped by regex and the last token of the remaining prefix is
located through the tokenizer's offset_mapping, giving the unpadded index of
the Y-final token (negative prompts) or the predicate-final token (positive
prompts). For Llama-3.1-8B and Mistral-7B-v0.1 this is a fixed offset from the
end of the prompt for every stem: -2 for stem / something_that / article, -4 for
question. Resolving through offset_mapping instead of hard-coding the offsets
means a new template only needs an entry in TEMPLATE_TAIL_PATTERNS.
"""

from __future__ import annotations

import re

# `something_that` / `article` tails are anchored to end-of-string so they do
# not match earlier `is` tokens inside the prompt (e.g. "Something that *is*
# an animal..." or "An animal that *is* not a reptile is").
TEMPLATE_TAIL_PATTERNS: dict[str, re.Pattern] = {
    "stem":           re.compile(r":\s*$"),
    "something_that": re.compile(r"\s+is\s*$", re.IGNORECASE),
    "article":        re.compile(r"\s+is\s*$", re.IGNORECASE),
    "question":       re.compile(r"\?\s*It\s*is\s*$", re.IGNORECASE),
}

# Template order of data/prompts-cleaned-release.json and
# data/prompts-cleaned-multi-release.json (blocks of 162 rows each).
TEMPLATE_INDEX_TO_ID: list[str] = [
    "stem",
    "something_that",
    "article",
    "question",
]

N_STEMS_PER_TEMPLATE = 162


def infer_template_id(prompt_number: int) -> str:
    """Recover the template id from a flat-file prompt_number.

    `prompts-cleaned-multi-release.json` has 648 rows numbered 0..647 with
    templates laid out in blocks of 162: 0..161 = stem, 162..323 =
    something_that, 324..485 = article, 486..647 = question.
    """
    block = prompt_number // N_STEMS_PER_TEMPLATE
    if not 0 <= block < len(TEMPLATE_INDEX_TO_ID):
        raise ValueError(
            f"prompt_number {prompt_number} is out of range for the 4-template layout"
        )
    return TEMPLATE_INDEX_TO_ID[block]


def resolve_target_token_idx(
    prompt: str,
    template_id: str,
    offset_mapping: list[tuple[int, int]],
) -> int:
    """Return the unpadded token index of the last "not Y" token (negative
    prompt) or of the predicate phrase (positive prompt).

    The function strips the per-template tail and then maps the new
    end-character to the token whose offset interval contains it. Special
    tokens (with offset `(0, 0)` or `(s, s)` for any `s`) are skipped.

    Raises ValueError if the tail does not match the prompt or if no token
    covers the resulting end character.
    """
    if template_id not in TEMPLATE_TAIL_PATTERNS:
        raise ValueError(
            f"Unknown template_id {template_id!r}. Known: {list(TEMPLATE_TAIL_PATTERNS)}"
        )
    tail_match = TEMPLATE_TAIL_PATTERNS[template_id].search(prompt)
    if tail_match is None:
        raise ValueError(
            f"Template {template_id!r} tail pattern did not match prompt: {prompt!r}"
        )
    end_excl = tail_match.start()
    # Strip any whitespace immediately before the tail so the end character
    # is a content character of "Y" rather than a space.
    while end_excl > 0 and prompt[end_excl - 1] in " \t":
        end_excl -= 1
    if end_excl <= 0:
        raise ValueError(f"Empty prefix after stripping tail for prompt: {prompt!r}")
    end_inclusive = end_excl - 1

    last_token_idx: int | None = None
    for i, (ts, te) in enumerate(offset_mapping):
        if ts == te:
            continue
        if ts <= end_inclusive < te:
            last_token_idx = i
            break
    if last_token_idx is None:
        raise ValueError(
            f"Could not locate token covering char {end_inclusive} for prompt {prompt!r} "
            f"with offsets {offset_mapping}"
        )
    return last_token_idx


def resolve_target_token_neg_idx(
    prompt: str,
    template_id: str,
    offset_mapping: list[tuple[int, int]],
    n_tokens: int,
) -> int:
    """Same as `resolve_target_token_idx` but expressed as a negative index
    relative to the end of the unpadded sequence (i.e. `target_idx - n_tokens`).

    Useful for a quick sanity log: the value should be -2 for stem /
    something_that / article and -4 for question.
    """
    return resolve_target_token_idx(prompt, template_id, offset_mapping) - n_tokens
