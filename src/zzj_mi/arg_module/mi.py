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
Command-line arguments shared by every compute script (parsed with HfArgumentParser).
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Argument:
    model_path: str = field(default="meta-llama/Llama-3.1-8B")
    model_revision: str = field(default="main", metadata={"help": "Hub revision; used for the OLMo-2 checkpoint sweep"})
    data_path: str = field(default="data/prompts-cleaned-release.json")
    output_path: str = field(default="outputs_release/")
    output_subdir: Optional[str] = field(default=None, metadata={"help": "optional extra directory under <output_path>/<model>/mi/"})

    batch_size: int = field(default=54)
    should_save: bool = field(default=False, metadata={"help": "write result files under the output path"})

    seed: int = field(default=42)
    disable_sanity_check: bool = field(default=False, metadata={"help": "skip printing the first batch's tokenization"})

    @property
    def normalized_model_path(self):
        return self.model_path.replace('/', '--')

    def get_complete_output_path(self):
        base_path = Path(self.output_path) / self.normalized_model_path / "mi"
        if self.output_subdir:
            return base_path / self.output_subdir
        return base_path

    def get_complete_output_path_with_revision(self):
        return Path(self.output_path) / self.normalized_model_path / self.model_revision / "mi"
