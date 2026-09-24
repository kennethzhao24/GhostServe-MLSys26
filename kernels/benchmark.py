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
    Benchmarks for prebuilt XOR, RDP, and RS CUDA extensions.
"""

import argparse
import torch

from function import (
    backwards_rdp_kv,
    backwards_rs_kv,
    backwards_xor_kv,
    forward_rdp_kv,
    forward_rs_kv,
    forward_xor_kv,
)
from utils import load_extension, set_seed, tensor_memory_size

def benchmark_xor(args):
    xor = load_extension("xor_kernel")
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")

    B, nh, T, hd = args.batch, args.heads, args.seq, args.head_dim
    keys = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16).contiguous()
    values = torch.randn(B, nh, T, hd, device=device, dtype=torch.float16).contiguous()
    print(f'Batch={B}, Heads={nh}, Seq={T}, HeadDim={hd}, N={B*nh*hd}')
    print(f'Size of each FP16 KV tensor: {tensor_memory_size(keys):.3f} GiB')

    torch.cuda.synchronize()

    pkg = forward_xor_kv(xor, keys, values, args.num_data_shards, args.num_parity_shards)

    keys_rt, values_rt, back_prof = backwards_xor_kv(xor, pkg, args.missing_shards)

    print("=== XOR Profiling (CUDA events) ===")
    print(f"Encode: {(pkg['encode_ms']):.3f} ms   ")
    print(f"Reconstruct:   {back_prof['reconstruct_ms']:.3f} ms")

    torch.cuda.synchronize()

def benchmark_rdp(args):
    rdp = load_extension("rdp_kernel")
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    shape = (args.batch, args.heads, args.seq, args.head_dim)
    keys = torch.randn(shape, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    print(f"Size of each FP16 KV tensor: {tensor_memory_size(keys):.3f} GiB")

    pkg = forward_rdp_kv(rdp, keys, values, args.num_data_shards,
                         args.num_parity_shards, args.star_base)
    keys_rt, values_rt, stats = backwards_rdp_kv(rdp, pkg, args.missing_shards)
    print("=== RDP Profiling (CUDA events) ===")
    print(f"Backup: {pkg['backup_ms']:.3f} ms")
    print(f"Recover: {stats['recover_ms']:.3f} ms")


def benchmark_rs(args):
    rs = load_extension("rs_kernel")
    set_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    shape = (args.batch, args.heads, args.seq, args.head_dim)
    keys = torch.randn(shape, device="cuda", dtype=torch.float16)
    values = torch.randn_like(keys)
    print(f"Size of each FP16 KV tensor: {tensor_memory_size(keys):.3f} GiB")

    pkg = forward_rs_kv(rs, keys, values, args.num_data_shards,
                        args.num_parity_shards)
    keys_rt, values_rt, stats = backwards_rs_kv(rs, pkg, args.missing_shards)
    print("=== RS Profiling (CUDA events) ===")
    print(f"Backup: {pkg['backup_ms']:.3f} ms")
    print(f"Recover: {stats['recover_ms']:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarks for prebuilt XOR, RDP, and RS CUDA extensions")
    parser.add_argument("--algorithm", choices=("xor", "rdp", "rs"), default="xor")
    parser.add_argument("--num_data_shards", type=int, default=8)
    parser.add_argument("--num_parity_shards", type=int,
                        help="Defaults to 1 for XOR and 2 for RDP/RS")
    parser.add_argument("--missing_shards", type=int, nargs="*",
                        help="Missing shard indices; defaults to 0 for XOR and 0,1 for RDP/RS")
    parser.add_argument("--star_base", type=int, default=0,
                        help="RDP diagonal parity offset")
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--heads",     type=int, default=8)
    parser.add_argument("--seq",       type=int, default=32768)
    parser.add_argument("--head_dim",  type=int, default=128)
    args = parser.parse_args()

    if args.num_parity_shards is None:
        args.num_parity_shards = 1 if args.algorithm == "xor" else 2
    if args.missing_shards is None:
        args.missing_shards = [0] if args.algorithm == "xor" else [0, 1]
    if args.algorithm == "rdp":
        args.missing_shards = [index for index in args.missing_shards if index != -1]
    if args.num_data_shards <= 0 or args.num_parity_shards < 0:
        parser.error("data shard count must be positive and parity shard count nonnegative")
    if args.algorithm == "xor" and args.num_parity_shards != 1:
        parser.error("XOR requires exactly one parity shard")
    if args.algorithm == "rdp" and args.num_parity_shards != 2:
        parser.error("RDP requires exactly two parity shards")
    if len(args.missing_shards) != len(set(args.missing_shards)):
        parser.error("missing shard indices must be distinct")
    max_index = (args.num_data_shards - 1 if args.algorithm == "rdp"
                 else args.num_data_shards + args.num_parity_shards - 1)
    if any(index < 0 or index > max_index for index in args.missing_shards):
        parser.error(f"missing shard indices must be in [0, {max_index}]")
    if args.algorithm == "xor" and len(args.missing_shards) > 1:
        parser.error("XOR can recover at most one missing shard")
    if args.algorithm == "rdp" and len(args.missing_shards) > 2:
        parser.error("RDP can recover at most two missing data shards")

    {"xor": benchmark_xor, "rdp": benchmark_rdp, "rs": benchmark_rs}[args.algorithm](args)
