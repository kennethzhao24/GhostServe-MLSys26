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


import torch
from utils import determine_shard_size, init_gf_tables

# ------------------------------ XOR Function ---------------------------------

def forward_xor_kv(xor_kernel, keys, values, num_data_shards, num_parity_shards):
    """
        Bit-cast FP16 K,V to bytes (IEEE-754 binary16 -> uint16 -> 2 bytes each),
        then XOR-encode into (num_data_shards + 1) shards.

        Byte layout: [K_bytes (T*n*2)] | [V_bytes (T*n*2)]
    """
    if num_parity_shards != 1:
        raise ValueError("XOR parity supports exactly 1 parity shard.")

    assert keys.is_cuda and values.is_cuda, "Run on CUDA for GPU timing."
    assert keys.dtype == torch.float16 and values.dtype == torch.float16
    assert keys.shape == values.shape

    # Ensure linear memory order matches our pack/unpack flattening
    keys   = keys.contiguous()
    values = values.contiguous()

    device = keys.device
    B, H, T, D = keys.shape
    n = B * H * D

    total_bytes = 4 * T * n  # 2 bytes per half, for K and V

    kv_bytes = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    xor_kernel.pack_kv(keys, values, kv_bytes, int(n), int(T))

    # --- XOR encode timing ---
    shard_size, pad = determine_shard_size(kv_bytes.numel(), num_data_shards)
    data_bytes = num_data_shards * shard_size

    data_padded = torch.zeros(data_bytes, dtype=torch.uint8, device=device)
    data_padded[:kv_bytes.numel()] = kv_bytes

    total_shards = num_data_shards + num_parity_shards
    shards_KV = torch.zeros((total_shards, shard_size), dtype=torch.uint8, device=device)

    e_start = torch.cuda.Event(enable_timing=True)
    e_end   = torch.cuda.Event(enable_timing=True)
    
    # warmup
    torch.cuda.synchronize()
    for _ in range(3):
        xor_kernel.encode_xor(data_padded, shards_KV, int(num_data_shards), int(shard_size))
    torch.cuda.synchronize()

    # benchmark
    torch.cuda.synchronize()
    e_start.record()
    for _ in range(10):
        xor_kernel.encode_xor(data_padded, shards_KV, int(num_data_shards), int(shard_size))
    e_end.record()
    torch.cuda.synchronize()
    encode_ms = e_start.elapsed_time(e_end)

    return {
        "shards_KV": shards_KV,
        "seq_len": T,
        "n": n,
        "shard_size": shard_size,
        "device": device,
        "shape_BHTD": (B, H, T, D),
        "encode_ms": encode_ms / 10,
        "num_data_shards": num_data_shards,
        "num_parity_shards": num_parity_shards,
        "pad": pad,
        "total_bytes": total_bytes,
        "orig_keys": keys,
        "orig_vals": values,
    }

def backwards_xor_kv(xor_kernel, pkg, missing_shards):
    """
        Reconstruct (if 1 missing shard), then unpack bytes back to fp16 K,V (bit-exact).
    """
    shards_KV        = pkg["shards_KV"]
    T                = pkg["seq_len"]
    n                = pkg["n"]
    shard_size       = pkg["shard_size"]
    device           = pkg["device"]
    B, H, _, D       = pkg["shape_BHTD"]
    num_data_shards  = pkg["num_data_shards"]
    num_parity_shards= pkg["num_parity_shards"]
    total_shards     = num_data_shards + num_parity_shards

    if num_parity_shards != 1:
        raise ValueError("XOR parity supports exactly 1 parity shard.")
    if len(missing_shards) > 1:
        raise ValueError("XOR parity can recover only a single missing shard.")
    for idx in missing_shards:
        if idx < 0 or idx >= total_shards:
            raise ValueError(f"Missing shard {idx} out of range 0..{total_shards-1}")

    # Zero the explicitly missing shard for clarity
    for idx in missing_shards:
        shards_KV[idx].zero_()

    r_ms = 0.0
    if len(missing_shards) == 1:
        missing_idx_tensor = torch.tensor(missing_shards, dtype=torch.int32, device=device)
        recovered_missing = torch.empty((1, shard_size), dtype=torch.uint8, device=device)
        r_start = torch.cuda.Event(enable_timing=True)
        r_end = torch.cuda.Event(enable_timing=True)

        # warmup
        torch.cuda.synchronize()
        for _ in range(3):
            xor_kernel.reconstruct_xor(shards_KV, shards_KV, recovered_missing,
            int(shard_size), int(num_data_shards), missing_idx_tensor
            )
        torch.cuda.synchronize()

        # benchmark
        torch.cuda.synchronize()
        r_start.record()
        for _ in range(10):
            xor_kernel.reconstruct_xor(
                shards_KV, shards_KV, recovered_missing,
                int(shard_size), int(num_data_shards), missing_idx_tensor
            )
        r_end.record()
        torch.cuda.synchronize()
        r_ms = r_start.elapsed_time(r_end)

    # Flatten data region back to byte stream
    byte_region = shards_KV[:num_data_shards].contiguous().view(-1)
    de_bytes_needed = 4 * T * n
    if byte_region.numel() < de_bytes_needed:
        raise RuntimeError("Not enough bytes in reconstructed data region.")

    reconstructed_KV = torch.empty(2 * T * n, dtype=torch.float16, device=device)

    xor_kernel.unpack_kv(byte_region, reconstructed_KV, int(n), int(T))


    keys_out   = reconstructed_KV[:T * n].view(B, H, T, D)
    values_out = reconstructed_KV[T * n : 2 * T * n].view(B, H, T, D)

    return keys_out, values_out, {
        "reconstruct_ms": r_ms / 10,
    }


# ------------------------------ RDP Function ---------------------------------

def forward_rdp_kv(rdp_kernel, keys, values, num_data_shards, num_parity_shards=2,
                   star_base=0):
    """Pack FP16 K/V and compute the two RDP parity shards."""
    if num_parity_shards != 2:
        raise ValueError("RDP requires exactly two parity shards.")
    if num_data_shards <= 0:
        raise ValueError("num_data_shards must be positive.")
    if not (keys.is_cuda and values.is_cuda):
        raise ValueError("K/V must be CUDA tensors.")
    if keys.dtype != torch.float16 or values.dtype != torch.float16:
        raise ValueError("K/V must be FP16 tensors.")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("K/V must have matching [B, H, T, D] shapes.")

    B, H, T, D = keys.shape
    n = B * H * D
    key_tn = keys.permute(2, 0, 1, 3).contiguous().view(T, n)
    value_tn = values.permute(2, 0, 1, 3).contiguous().view(T, n)
    shard_size, pad_words = determine_shard_size(2 * T * n, num_data_shards)
    shards = torch.zeros((num_data_shards + 2, shard_size),
                         dtype=torch.uint16, device=keys.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    rdp_kernel.fused_backup_full_kv(
        key_tn, value_tn, shards, num_data_shards, shard_size, T, n, star_base
    )
    end.record()
    end.synchronize()

    return {
        "shards_KV": shards,
        "shard_size": shard_size,
        "seq_len": T,
        "n": n,
        "num_data_shards": num_data_shards,
        "star_base": star_base,
        "shape_BHTD": (B, H, T, D),
        "pad_words": pad_words,
        "backup_ms": start.elapsed_time(end),
    }


def backwards_rdp_kv(rdp_kernel, pkg, missing_shards):
    """Recover up to two missing RDP data shards and unpack FP16 K/V."""
    nd = pkg["num_data_shards"]
    missing = [index for index in missing_shards if index != -1]
    if len(missing) > 2 or len(missing) != len(set(missing)):
        raise ValueError("RDP requires at most two distinct missing data shards.")
    if any(index < 0 or index >= nd for index in missing):
        raise ValueError(f"RDP missing shard indices must be in [0, {nd - 1}].")

    shards = pkg["shards_KV"]
    for index in missing:
        shards[index].zero_()

    T, n = pkg["seq_len"], pkg["n"]
    B, H, _, D = pkg["shape_BHTD"]
    packed_tn = torch.empty(2 * T * n, dtype=torch.float16, device=shards.device)
    first = missing[0] if missing else -1
    second = missing[1] if len(missing) == 2 else -1

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    rdp_kernel.fused_recovery_full_kv(
        shards, packed_tn, nd, pkg["shard_size"], T, n, pkg["star_base"],
        first, second
    )
    end.record()
    end.synchronize()

    keys = packed_tn[:T * n].view(T, B, H, D).permute(1, 2, 0, 3).contiguous()
    values = packed_tn[T * n:].view(T, B, H, D).permute(1, 2, 0, 3).contiguous()
    return keys, values, {"recover_ms": start.elapsed_time(end)}


# ------------------------------ Reed-Solomon Function ---------------------------------

def forward_rs_kv(rs_kernel, keys, values, num_data_shards, num_parity_shards):
    """Pack FP16 K/V into uint16 data shards and compute RS parity shards."""
    if num_data_shards <= 0 or num_parity_shards < 0:
        raise ValueError("RS requires positive data and nonnegative parity shard counts.")
    if not (keys.is_cuda and values.is_cuda):
        raise ValueError("K/V must be CUDA tensors.")
    if keys.dtype != torch.float16 or values.dtype != torch.float16:
        raise ValueError("K/V must be FP16 tensors.")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("K/V must have matching [B, H, T, D] shapes.")

    B, H, T, D = keys.shape
    n_k, n_v = keys.numel(), values.numel()
    total = n_k + n_v
    shard_size, pad_words = determine_shard_size(total, num_data_shards)
    packed = torch.empty(total, dtype=torch.uint16, device=keys.device)
    data_padded = torch.zeros(num_data_shards * shard_size,
                              dtype=torch.uint16, device=keys.device)
    shards = torch.empty((num_data_shards + num_parity_shards, shard_size),
                         dtype=torch.uint16, device=keys.device)
    gf_exp_np, gf_log_np = init_gf_tables()
    gf_exp = torch.from_numpy(gf_exp_np).to(device=keys.device)
    gf_log = torch.from_numpy(gf_log_np).to(device=keys.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    rs_kernel.pack_fp16_pair_to_u16(
        keys.contiguous().view(-1), values.contiguous().view(-1), packed
    )
    data_padded[:total].copy_(packed)
    rs_kernel.encode_rs_u16(
        data_padded, shards, num_data_shards, num_parity_shards,
        shard_size, gf_exp, gf_log
    )
    end.record()
    end.synchronize()

    return {
        "shards_KV": shards,
        "shape_BHTD": (B, H, T, D),
        "num_data_shards": num_data_shards,
        "num_parity_shards": num_parity_shards,
        "shard_size": shard_size,
        "pad_words": pad_words,
        "total_words": total,
        "n_k": n_k,
        "n_v": n_v,
        "gf_exp": gf_exp,
        "gf_log": gf_log,
        "backup_ms": start.elapsed_time(end),
    }


def backwards_rs_kv(rs_kernel, pkg, missing_shards):
    """Recover missing RS data shards and unpack FP16 K/V."""
    nd = pkg["num_data_shards"]
    np = pkg["num_parity_shards"]
    missing = list(missing_shards)
    if len(missing) != len(set(missing)):
        raise ValueError("RS missing shard indices must be distinct.")
    if any(index < 0 or index >= nd + np for index in missing):
        raise ValueError(f"RS missing shard indices must be in [0, {nd + np - 1}].")
    missing_data = sorted(index for index in missing if index < nd)
    available_parity = [index for index in range(np) if nd + index not in missing]
    if len(missing_data) > 16:
        raise ValueError("RS supports at most 16 missing data shards.")
    if len(missing_data) > len(available_parity):
        raise ValueError("Not enough parity shards to recover the missing data shards.")

    shards = pkg["shards_KV"]
    for index in missing:
        shards[index].zero_()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if missing_data:
        missing_data_t = torch.tensor(missing_data, dtype=torch.int32,
                                      device=shards.device)
        parity_used_t = torch.tensor(available_parity[:len(missing_data)],
                                     dtype=torch.int32, device=shards.device)
        rs_kernel.reconstruct_rs_u16_inplace(
            shards, pkg["shard_size"], nd, np,
            missing_data_t, parity_used_t, pkg["gf_exp"], pkg["gf_log"]
        )

    data = shards[:nd].contiguous().view(-1)[:pkg["total_words"]]
    keys = torch.empty(pkg["n_k"], dtype=torch.float16, device=shards.device)
    values = torch.empty(pkg["n_v"], dtype=torch.float16, device=shards.device)
    rs_kernel.u16_to_fp16_split(data, keys, values, pkg["n_k"], pkg["n_v"])
    end.record()
    end.synchronize()

    shape = pkg["shape_BHTD"]
    return keys.view(shape), values.view(shape), {"recover_ms": start.elapsed_time(end)}
