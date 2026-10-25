"""stress_mem.py --seconds S [--gb G] — banda e integridade da GDDR6X para o degrau de offset de memória.

A cada iteração: gera no GPU um padrão determinístico novo (semente = número da iteração) em `src`, copia para `dst`
(a banda vem desta cópia, medida com eventos CUDA: lê e escreve G/2 GB cada) e confere `dst` contra o padrão gerado de
novo. Qualquer diferença é erro de memória (célula ou cópia). Erros de enlace a GDDR6X retransmite (EDR): não corrompem,
mas derrubam a banda; por isso a banda por degrau importa tanto quanto a integridade.
Uma linha JSON a cada ~5 s; sai com 3 na primeira divergência."""
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
        for _ in range(COPIES // 2):   # vai e volta: a banda vem só das cópias, sem a geração do padrão (que pesa no PL)
            ev0.record()
            dst.copy_(src)
            src.copy_(dst)
            ev1.record()
            torch.cuda.synchronize()
            bws.append(4 * src.numel() * 8 / (ev0.elapsed_time(ev1) / 1000) / 1e9)
            mem.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            ws.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000)
        pattern(src, it)          # regenera: corrupção em qualquer cópia aparece na comparação com dst
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
