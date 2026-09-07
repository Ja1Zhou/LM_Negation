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

# src/zzj_mi/olmo_module/generate_download_commands_from_checkpoints.py
import re
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "allenai/OLMo-2-1124-7B"
STEP_PATTERN = re.compile(r"stage1-step(\d+)-")
CHECKPOINTS_PATH = Path(__file__).with_name("checkpoints.txt")


def parse_branch_step(branch_name: str) -> int | None:
    match = STEP_PATTERN.search(branch_name)
    if not match:
        return None
    return int(match.group(1))


def read_target_steps(path: Path) -> list[int]:
    steps: list[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        steps.append(int(line))
    return steps


def generate_download_cmd(revision: str) -> str:
    return f"hf download {REPO_ID} --revision {revision}"


def main() -> None:
    api = HfApi()
    refs = api.list_repo_refs(REPO_ID)
    stage1_branches = [b.name for b in refs.branches if b.name.startswith("stage1-")]

    step_to_branch: dict[int, str] = {}
    for branch in stage1_branches:
        step = parse_branch_step(branch)
        if step is None:
            continue
        step_to_branch[step] = branch

    target_steps = read_target_steps(CHECKPOINTS_PATH)

    missing_steps: list[int] = []
    for step in target_steps:
        branch = step_to_branch.get(step)
        if branch is None:
            missing_steps.append(step)
            continue
        print(generate_download_cmd(branch))

    if missing_steps:
        print("\n# Missing steps (no matching stage1 branch):")
        for step in missing_steps:
            print(f"# {step}")


if __name__ == "__main__":
    main()
