"""stress_vf.py --seconds S --lock MHZ [--target-w W] — stability of one V/F curve point with the tensor unlock active.

BF16 GEMM with a closed-loop duty cycle that keeps the GPU at the lock clock and close to --target-w: the highest
current the lock point sees under the power limit, which is the serving regime. A plain dense GEMM is useless here:
the power limiter drops it to ~600 MHz, at the minimum voltage, and the lock point is never exercised.
The loop only lowers the load when the clock dropped because of power; with too little load it raises it (the first
version lowered it in both cases and, at 1800 MHz on the 3090s, spiralled down to a 0.05 duty).
Every result is compared bit for bit with the first one (same inputs, same kernel, deterministic cuBLAS): any
difference is instability. One JSON line every ~5 s; exits with 3 on the first mismatch."""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

try:
    import pynvml
except ImportError:  # pragma: no cover
    from vllm.third_party import pynvml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--lock", type=int, required=True, help="maximum clock of -lgc, in MHz")
    p.add_argument("--target-w", type=float, default=205)
    p.add_argument("--size", type=int, default=4096)
    a = p.parse_args()
    torch.use_deterministic_algorithms(True)
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    g = torch.Generator(device="cuda").manual_seed(1234)
    A = torch.randn(a.size, a.size, device="cuda", dtype=torch.bfloat16, generator=g)
    B = torch.randn(a.size, a.size, device="cuda", dtype=torch.bfloat16, generator=g)
    ref = A @ B
    torch.cuda.synchronize()
    duty, period, iters, bad = 0.5, 0.05, 0, 0
    t0 = last_adj = last_out = time.monotonic()
    mhz_w = []
    print(json.dumps({"start": True, "lock": a.lock, "target_w": a.target_w, "size": a.size}), flush=True)
    while time.monotonic() - t0 < a.seconds:
        busy_end = time.monotonic() + duty * period
        while time.monotonic() < busy_end:
            C = A @ B
            if not torch.equal(C, ref):
                bad += 1
            iters += 1
        torch.cuda.synchronize()
        if duty < 1:
            time.sleep((1 - duty) * period)
        now = time.monotonic()
        mhz = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
        mhz_w.append((mhz, w))
        if now - last_adj >= 0.5:   # loop: stay at the lock, as close to target-w as possible
            if mhz < a.lock - 30 and w >= a.target_w - 8:
                duty = max(0.05, duty * 0.9)      # clock dropped because of power: less load
            elif w < a.target_w - 8:
                duty = min(1.0, duty * 1.06)      # power headroom (or low clock from too little load): more load
            last_adj = now
        if bad:
            print(json.dumps({"t_s": round(now - t0, 1), "MISMATCH": bad, "iters": iters, "mhz": mhz, "w": w}), flush=True)
            sys.exit(3)
        if now - last_out >= 5:
            ms = sorted(x[0] for x in mhz_w)
            ws = sorted(x[1] for x in mhz_w)
            print(json.dumps({"t_s": round(now - t0, 1), "iters": iters, "mismatch": bad, "duty": round(duty, 3),
                              "mhz_p50": ms[len(ms) // 2], "w_p50": round(ws[len(ws) // 2], 1),
                              "w_max": round(ws[-1], 1)}), flush=True)
            mhz_w.clear()
            last_out = now
    print(json.dumps({"done": True, "iters": iters, "mismatch": bad}), flush=True)


if __name__ == "__main__":
    main()
