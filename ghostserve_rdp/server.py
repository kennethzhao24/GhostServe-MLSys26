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
    SGLang server launch, hook staging, readiness, and shutdown.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests


def _stage_hook(tmpdir: str) -> str:
    source_dir = Path(__file__).parent / "hook"
    for filename in ("sitecustomize.py", "rdp_kv_ops.cu"):
        shutil.copy2(source_dir / filename, Path(tmpdir) / filename)
    return str(Path(tmpdir) / "sitecustomize.py")


# ---------------------- server orchestration ----------------------

def launch_sglang_server(
    model_id: str,
    tp: int,
    chunk_size: int,
    port: int,
    host: str,
    gpus: str,
    python_bin: str,
    extra_args: Optional[List[str]],
    kv_cache_dtype: str,
    download_dir: Optional[str],
    gather_kv: bool,
    prog_start_mono: float,
    metrics_file: str,
) -> Tuple[subprocess.Popen, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")

    # Prefer NVLink/NVSwitch intra-node paths when present
    env.setdefault("NCCL_P2P_LEVEL", "NVL")
    # If you rely on NCCL-IB, remove/override this
    env.setdefault("NCCL_IB_DISABLE", "1")

    env["SGLANG_GATHER_KV_TO_GPU0"] = "1" if gather_kv else "0"
    env.setdefault("SGLANG_SNAPSHOT_WEIGHTS", "0")
    env.setdefault("SGLANG_SIMULATE_GPU01_FAIL", "0")
    env.setdefault("RDP_STAR_BASE", "0")

    tmpdir = tempfile.mkdtemp(prefix="kvhook_rdp_")
    sc_path = _stage_hook(tmpdir)
    metrics_path = Path(metrics_file).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    metrics_file = str(metrics_path)

    env["PYTHONPATH"] = f"{tmpdir}:{env.get('PYTHONPATH','')}"
    env["FT_METRICS_FILE"] = metrics_file
    env["FT_PROG_START_MONO"] = f"{prog_start_mono:.9f}"

    print(f"[info] Injected sitecustomize at: {sc_path}")
    print(f"[info] Server metrics file: {metrics_file}")

    cmd = [
        python_bin, "-m", "sglang.launch_server",
        "--model", model_id,
        "--tensor-parallel-size", str(tp),
        "--chunked-prefill-size", str(chunk_size),
        "--enable-mixed-chunk",
        "--dtype", "bfloat16",
        "--kv-cache-dtype", kv_cache_dtype,
        "--host", host, "--port", str(port),
        "--trust-remote-code",
    ]
    if download_dir:
        cmd += ["--download-dir", download_dir]
    if extra_args:
        cmd.extend(extra_args)

    print("[info] Launching SGLang server:\n       " + " ".join(cmd))
    print(f"[info] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump(p):
        try:
            for line in iter(p.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write("[sglang] " + line)
        except Exception:
            pass

    threading.Thread(target=_pump, args=(proc,), daemon=True).start()
    return proc, metrics_file


def wait_until_ready(base_url: str, timeout_s: int = 900) -> None:
    t0 = time.time()
    last_error = None
    print(f"[info] Waiting for server ready at {base_url} ...")
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/models", timeout=5)
            if r.status_code == 200:
                print("[info] Server is ready.")
                return
            last_error = f"HTTP {r.status_code} {r.text[:120]}"
        except Exception as e:
            last_error = str(e)
        time.sleep(2)
    raise RuntimeError(f"SGLang server did not become ready in {timeout_s}s. Last error: {last_error}")


def terminate_process(proc: Optional[subprocess.Popen], grace_s: float = 10.0):
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass


