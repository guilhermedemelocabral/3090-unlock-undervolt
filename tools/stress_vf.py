"""stress_vf.py --seconds S --lock MHZ [--target-w W] — estabilidade de um ponto da curva V/F com o unlock ativo.

GEMM BF16 com ciclo de trabalho ajustado em malha fechada para a GPU ficar no clock do lock e perto de --target-w:
a corrente máxima que o ponto do lock vê sob o power limit, que é o regime do serving.
A malha só reduz a carga quando o clock caiu por potência; com pouca carga ela aumenta (a 1ª versão
reduzia nos dois casos e, a 1800 MHz nas 3090, entrava numa espiral até duty 0,05). O GEMM denso puro não serve,
porque o limitador o derruba para ~600 MHz, na voltagem mínima, e o ponto do lock nem é exercitado.
Cada resultado é comparado bit a bit com o primeiro (mesmas entradas, mesmo kernel, cuBLAS determinístico):
qualquer diferença é instabilidade. Uma linha JSON a cada ~5 s; sai com 3 na primeira divergência."""
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
    p.add_argument("--lock", type=int, required=True, help="clock máximo do -lgc, em MHz")
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
        if now - last_adj >= 0.5:   # malha: no lock e o mais perto possível de target-w
            if mhz < a.lock - 30 and w >= a.target_w - 8:
                duty = max(0.05, duty * 0.9)      # clock caiu por potência: menos carga
            elif w < a.target_w - 8:
                duty = min(1.0, duty * 1.06)      # sobra potência (ou clock baixo por falta de carga): mais carga
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
