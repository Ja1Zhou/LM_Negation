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

uv venv clean_env --no-project -p 3.12 --seed --relocatable
source scripts/setup_env.sh
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==4.57.3
uv pip install openai==2.21.0 python-dotenv==1.2.1   # OpenRouter client for src/zzj_mi/annotate (figure5.sh)

git clone https://github.com/TransformerLensOrg/TransformerLens.git vendor/TransformerLens
cd vendor/TransformerLens 
git fetch origin pull/816/head:OLMo
git checkout OLMo
git checkout 9febc5cc
sed -i '/"Qwen\/Qwen3-8B",/a\    "Qwen/Qwen3-8B-Base", # this supports the Qwen3-8B-Base model' transformer_lens/loading_from_pretrained.py
uv pip install -e .

cd ../../
uv pip install -e .

python -c "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformer_lens import HookedTransformer; print('TL import ok')"