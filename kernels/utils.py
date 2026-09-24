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
    Utilities for prebuilt XOR, RDP and RS CUDA extensions.
"""

import importlib.util
import random
import sysconfig

from pathlib import Path

import numpy as np

import torch

GF_SIZE = 1 << 16
GF_ORDER = GF_SIZE - 1


def load_extension(name):
    kernel_dir = Path(__file__).parent
    path = kernel_dir / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    if not path.is_file():
        path = kernel_dir / "sm90" / (name + ".so")
    if not path.is_file():
        raise ImportError(f"Could not find CUDA extension: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CUDA extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_memory_size(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024 ** 3)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def gbps(total_bytes: int, ms: float) -> float:
    if ms <= 0:
        return float("inf")
    return (total_bytes / 1e9) / (ms / 1e3)


def determine_shard_size(num_bytes: int, num_data_shards: int):
    if num_data_shards <= 0:
        raise ValueError("num_data_shards must be positive")
    pad = (num_data_shards - (num_bytes % num_data_shards)) % num_data_shards
    padded_total = num_bytes + pad
    shard_size = padded_total // num_data_shards
    return shard_size, pad


def init_gf_tables():
    # Primitive polynomial for GF(2^16). Keep consistent across encode/decode.
    primitive_polynomial = 0x1100B
    gf_log = np.zeros(GF_SIZE, dtype=np.uint16)
    gf_exp = np.zeros(2 * GF_ORDER, dtype=np.uint16)

    x = 1
    for i in range(GF_ORDER):
        gf_exp[i] = x
        gf_log[x] = i
        x <<= 1
        if x & GF_SIZE:  # x >= 2^16
            x ^= primitive_polynomial

    gf_exp[GF_ORDER:] = gf_exp[:GF_ORDER]
    return gf_exp, gf_log
