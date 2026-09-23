# Copyright (c) 2026 Kenneth Zhao

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
    Model configuration and KV cache sizing.
"""

from dataclasses import dataclass
from typing import Dict


# ---------------------- KV sizing helpers ----------------------

@dataclass
class ModelSpec:
    layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads


FALLBACK_SPECS: Dict[str, ModelSpec] = {
    "meta-llama/Llama-3.1-8B-Instruct":  ModelSpec(layers=32, n_heads=32, n_kv_heads=8,  hidden_size=4096),
    "meta-llama/Llama-3.1-70B-Instruct": ModelSpec(layers=80, n_heads=64, n_kv_heads=8,  hidden_size=8192),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ModelSpec(layers=64, n_heads=40, n_kv_heads=8, hidden_size=5120),
    "openai/gpt-oss-120b":               ModelSpec(layers=36, n_heads=64, n_kv_heads=8,  hidden_size=4096),
}


def load_model_spec(model_id: str) -> ModelSpec:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return ModelSpec(
            layers=int(getattr(cfg, "num_hidden_layers")),
            n_heads=int(getattr(cfg, "num_attention_heads")),
            n_kv_heads=int(getattr(cfg, "num_key_value_heads", int(getattr(cfg, "num_attention_heads")))),
            hidden_size=int(getattr(cfg, "hidden_size")),
        )
    except Exception as e:
        print(f"[warn] AutoConfig load failed for '{model_id}': {e}")
        if model_id in FALLBACK_SPECS:
            print("[info] Using fallback spec.")
            return FALLBACK_SPECS[model_id]
        raise


def kv_bytes(n_layers: int, n_kv_heads: int, head_dim: int, tokens: int, bytes_per_elem: int = 2) -> int:
    # K and V
    return n_layers * tokens * (2 * n_kv_heads * head_dim) * bytes_per_elem


def mib(x: int) -> float:
    return x / (1024 ** 2)


def gib(x: int) -> float:
    return x / (1024 ** 3)


