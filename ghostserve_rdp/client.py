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
    Streamed inference and time-to-first-token measurement.
"""

import threading
import time
from typing import List, Optional

from openai import OpenAI


# ---------------------- simple TTFT client ----------------------

def _ttft_one(base_url: str, api_key: str, prompt: str, max_new_tokens: int,
              start_gate: threading.Barrier, idx: int, out: List[float]):
    client = OpenAI(base_url=base_url, api_key=api_key)
    start_gate.wait()
    t0 = time.perf_counter()
    first_ms = None
    stream = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.0,
        max_tokens=max_new_tokens,
    )
    for event in stream:
        if getattr(event, "choices", None):
            delta = event.choices[0].delta
            if (delta and (delta.content or delta.role)) and first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000.0
                if idx == 0 and getattr(delta, "content", None):
                    print(delta.content, end="", flush=True)
    if idx == 0:
        print()
    out[idx] = first_ms if first_ms is not None else float("nan")


def measure_ttft_batch(base_url: str, api_key: str, prompt: str,
                       batch_size: int, max_new_tokens: int = 64,
                       input_tokens: Optional[int] = None,
                       model_id: Optional[str] = None) -> List[float]:
    if input_tokens is not None:
        assert model_id is not None, "model_id is required when input_tokens is set"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        base = "hello "
        base_ids = tok.encode(base, add_special_tokens=False)
        rep = max(1, (input_tokens // max(1, len(base_ids))) + 1)
        text = base * rep
        ids = tok.encode(text, add_special_tokens=False)[:input_tokens]
        prompt = tok.decode(ids)

    out: List[float] = [float("nan")] * batch_size
    gate = threading.Barrier(parties=batch_size)
    threads = []
    for i in range(batch_size):
        t = threading.Thread(target=_ttft_one, args=(base_url, api_key, prompt, max_new_tokens, gate, i, out))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return out


