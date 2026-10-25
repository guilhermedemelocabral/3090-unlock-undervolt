"""stress_mem.py --seconds S [--gb G] — GDDR6X bandwidth and integrity, for the memory-offset steps.

Each iteration generates a new deterministic pattern on the GPU (seed = iteration number) in `src`, copies it back and
forth to `dst` (bandwidth comes only from these copies, timed with CUDA events) and checks `dst` against the pattern
generated again. Any difference is a memory error (cell or copy). GDDR6X retransmits link errors (EDR): they do not
corrupt data but they lower bandwidth, so bandwidth per step matters as much as integrity.
One JSON line every ~5 s; exits with 3 on the first mismatch."""
import argparse
import json
import statistics
import sys
import time

import torch

try:
    import pynvml
except ImportError:  # pragma: no cover
    from vllm.third_party import pynvml


COPIES = 10


def pattern(out, seed):
    torch.arange(out.numel(), device=out.device, dtype=torch.int64, out=out)
    out.mul_(6364136223846793005).add_(1442695040888963407 + seed * 2654435761)
    out.bitwise_xor_(out >> 29)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--gb", type=float, default=10)
    a = p.parse_args()
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    n = int(a.gb * 2**30 / 8 / 2)
    src = torch.empty(n, dtype=torch.int64, device="cuda")
    dst = torch.empty_like(src)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print(json.dumps({"start": True, "gb": a.gb}), flush=True)
    t0 = last = time.monotonic()
    it, bad, bws, mem, ws = 0, 0, [], [], []
    while time.monotonic() - t0 < a.seconds:
        pattern(src, it)
        for _ in range(COPIES // 2):   # back and forth: bandwidth comes only from the copies, not from the pattern generation (which loads the PL)
            ev0.record()
            dst.copy_(src)
            src.copy_(dst)
            ev1.record()
            torch.cuda.synchronize()
            bws.append(4 * src.numel() * 8 / (ev0.elapsed_time(ev1) / 1000) / 1e9)
            mem.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            ws.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000)
        pattern(src, it)          # regenerate: corruption in any copy shows up in the comparison with dst
        if not torch.equal(dst, src):
            bad += 1
            print(json.dumps({"t_s": round(time.monotonic() - t0, 1), "MISMATCH": bad, "iter": it,
                              "diff": int((dst != src).sum())}), flush=True)
            sys.exit(3)
        it += 1
        if time.monotonic() - last >= 5:
            print(json.dumps({"t_s": round(time.monotonic() - t0, 1), "iters": it, "mismatch": 0,
                              "bw_gbs_p50": round(statistics.median(bws), 1), "bw_gbs_min": round(min(bws), 1),
                              "mem_mhz": statistics.median(mem), "w_p50": round(statistics.median(ws), 1)}), flush=True)
            bws, mem, ws, last = [], [], [], time.monotonic()
    print(json.dumps({"done": True, "iters": it, "mismatch": bad}), flush=True)


if __name__ == "__main__":
    main()
