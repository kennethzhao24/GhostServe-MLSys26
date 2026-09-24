# GhostServe
[GhostServe](https://proceedings.mlsys.org/paper_files/paper/2026/file/d99e8e80a6c41e148db686918dd7eab3-Paper-Conference.pdf) is a lightweight checkpointing protocol for KV cache in LLM serving systems. It operates based on the classical idea of erasure coding, where it generates additional parity KV cache stored in host memory to protect distributed GPU KV cache. This code provides a demo implementation of GhostServe for SGLang v.0.5.13.

## Installation

```bash
```


## Run

Set the Hugging Face token, then start the server in one terminal:

```bash
export HF_TOKEN="YOUR_TOKEN"
python launch_ghostserve_rdp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp 8 --chunk-size 2048 \
  --kv-cache-dtype auto \
  --download-dir /work/nvme/bfgy/llm-models \
  --gather-kv true \
  --metrics-file ./ft_metrics.jsonl
```

In another terminal, measure streamed time to first token:

```bash
python bench_ghostserve_rdp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --chunk-size 2048 --batch-size 16 \
  --kv-cache-dtype auto --input-tokens 32768 \
  --metrics-file ./ft_metrics.jsonl
```

Both commands use `./ft_metrics.jsonl` by default. Pass the same explicit path when
running them from different working directories. The server clears this file on
startup; the benchmark prints its latest RDP metrics record after measuring TTFT.