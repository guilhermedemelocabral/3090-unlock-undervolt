#!/usr/bin/env python3
"""GEMM 8192^3 por tipo, com TFLOPS por MHz de SM.

O número que decide é TFLOPS/MHz: ele não depende do limite de potência, que
derruba o clock quando o tensor core passa a render mais por ciclo.
Referência GA102 (relatos do gpu-unlock, issues #1 e #5):
  BF16 e FP16 acumulando em FP32: ~0,043 TFLOPS/MHz travado, ~0,083 destravado
  TF32: metade disso
  FP16 acumulando em FP16: ~0,083 sempre (controle de taxa cheia)

Uso, uma GPU por vez, numa imagem com PyTorch e pynvml (nos testes, a do vLLM):
  docker run --rm --gpus '"device=0"' -v "$PWD":/bench:ro --entrypoint python3 \
    vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c /bench/bench_gemm.py
Soak (BF16 contínuo, uma linha JSON a cada ~10 s): acrescente --soak 600.
"""
import argparse
import json
import statistics
import sys
import threading
import time

import torch

try:
    import pynvml
except ImportError:
    try:
        from vllm.third_party import pynvml
    except ImportError:
        pynvml = None

HANDLE = None
if pynvml is not None:
    try:
        pynvml.nvmlInit()
        # Dentro do container só a GPU escolhida é visível.
        HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        HANDLE = None

N = 8192
FLOP_PER_GEMM = 2 * N**3

CASES = [
    # nome, dtype, tf32, fp16_acc
    ("bf16_acc32", torch.bfloat16, False, False),
    ("fp16_acc32", torch.float16, False, False),
    ("tf32", torch.float32, True, False),
    ("fp16_acc16_controle", torch.float16, False, True),
]


def text(value):
    return value.decode() if isinstance(value, bytes) else value


class Sampler(threading.Thread):
    """Amostra clock de SM e potência a cada 50 ms enquanto a medição roda."""

    def __init__(self):
        super().__init__(daemon=True)
        self.mhz = []
        self.watts = []
        self.done = threading.Event()

    def run(self):
        while not self.done.is_set():
            if HANDLE is not None:
                self.mhz.append(pynvml.nvmlDeviceGetClockInfo(HANDLE, pynvml.NVML_CLOCK_SM))
                self.watts.append(pynvml.nvmlDeviceGetPowerUsage(HANDLE) / 1000)
            time.sleep(0.05)

    def stop(self):
        self.done.set()
        self.join()
        mhz = statistics.fmean(self.mhz) if self.mhz else float("nan")
        watts = statistics.fmean(self.watts) if self.watts else float("nan")
        return mhz, watts


def set_mode(tf32, fp16_acc):
    """Devolve False quando o torch não sabe acumular FP16 em FP16."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_acc
        return True
    return not fp16_acc


def timed(a, b, c, iters):
    sampler = Sampler()
    sampler.start()
    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    mhz, watts = sampler.stop()
    return FLOP_PER_GEMM * iters / seconds / 1e12, mhz, watts


def operands(dtype):
    a = torch.randn(N, N, device="cuda", dtype=dtype)
    b = torch.randn(N, N, device="cuda", dtype=dtype)
    return a, b, torch.empty(N, N, device="cuda", dtype=dtype)


def header():
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
    }
    if HANDLE is not None:
        info["pci"] = text(pynvml.nvmlDeviceGetPciInfo(HANDLE).busId)
        info["driver"] = text(pynvml.nvmlSystemGetDriverVersion())
        info["vbios"] = text(pynvml.nvmlDeviceGetVbiosVersion(HANDLE))
        info["power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(HANDLE) / 1000
    else:
        info["nvml"] = "indisponível: sem clock nem potência, só TFLOPS"
    return info


def bench(reps, iters, warmup):
    print(json.dumps({"header": header()}), flush=True)
    per_mhz_by_case = {}
    for name, dtype, tf32, fp16_acc in CASES:
        if not set_mode(tf32, fp16_acc):
            print(json.dumps({"case": name, "skipped": "torch sem allow_fp16_accumulation"}), flush=True)
            continue
        a, b, c = operands(dtype)
        for _ in range(warmup):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()
        runs = [timed(a, b, c, iters) for _ in range(reps)]
        per_mhz = [tflops / mhz for tflops, mhz, _ in runs if mhz > 0]  # NaN falha o > 0
        row = {
            "case": name,
            "tflops_median": round(statistics.median([r[0] for r in runs]), 2),
            "sm_mhz_median": round(statistics.median([r[1] for r in runs]), 1),
            "watts_median": round(statistics.median([r[2] for r in runs]), 1),
            "tflops_per_mhz_median": round(statistics.median(per_mhz), 4) if per_mhz else None,
            "runs_tflops_mhz_watts": [[round(x, 2) for x in r] for r in runs],
        }
        per_mhz_by_case[name] = row["tflops_per_mhz_median"]
        print(json.dumps(row), flush=True)
        del a, b, c
        torch.cuda.empty_cache()
    set_mode(False, False)

    bf16 = per_mhz_by_case.get("bf16_acc32")
    if bf16 is None:
        verdict = "sem clock de SM: compare TFLOPS no mesmo limite de potência"
    elif bf16 < 0.06:
        verdict = "BF16 em meia taxa (travado)"
    else:
        verdict = "BF16 em taxa cheia (destravado)"
    print(f"BF16 TFLOPS/MHz = {bf16} -> {verdict}", file=sys.stderr)


def soak(seconds):
    print(json.dumps({"header": header(), "soak_s": seconds}), flush=True)
    set_mode(False, False)
    a, b, c = operands(torch.bfloat16)
    tflops, _, _ = timed(a, b, c, 20)
    iters = max(10, int(10 * tflops * 1e12 / FLOP_PER_GEMM))  # janelas de ~10 s
    begin = time.monotonic()
    while time.monotonic() - begin < seconds:
        tflops, mhz, watts = timed(a, b, c, iters)
        print(json.dumps({
            "t_s": round(time.monotonic() - begin),
            "tflops": round(tflops, 2),
            "sm_mhz": round(mhz, 1),
            "watts": round(watts, 1),
            "tflops_per_mhz": round(tflops / mhz, 4) if mhz > 0 else None,
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--soak", type=int, default=0, help="segundos de carga BF16 contínua")
    args = parser.parse_args()
    if args.soak:
        soak(args.soak)
    else:
        bench(args.reps, args.iters, args.warmup)


if __name__ == "__main__":
    main()
