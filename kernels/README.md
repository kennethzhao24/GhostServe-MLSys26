# Erasure Coding Kernels (CUDA)

This directory contains Python wrappers and a benchmark entry point for prebuilt
XOR, RDP, and Reed–Solomon (RS) CUDA extensions, built on Hopper GPUs with CUDA 13.0.


```bash
python benchmark.py \
   --algorithm xor \
   --num_data_shards 8 \
   --num_parity_shards 1 \
   --missing_shards 0 \
   --batch 1 \
   --heads 8 \
   --seq 32678 \
   --head_dim 128
```
