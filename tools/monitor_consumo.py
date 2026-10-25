"""monitor_consumo.py [--resto-w W] [--fonte-w W] [--csv ARQ] — consumo do sistema, uma linha por segundo.

Amostra a cada 100 ms:
- cada GPU: potência instantânea da placa inteira (NVML, campo 186);
- CPU: energia do pacote (RAPL).
Por linha imprime a média e o pico do segundo de cada GPU, a soma das GPUs (média e pico da soma instantânea),
a CPU, o total medido e o total estimado. O estimado soma --resto-w (placa-mãe, RAM, discos e fans; NÃO medido,
padrão 60 W) e divide por 0,90 de eficiência da fonte para estimar a tomada. Com --fonte-w, mostra a carga em % da
fonte. "pico medido" é o maior total medido (GPUs + CPU, pico) desde o início.
Tensões da fonte pelo Super I/O da placa-mãe (NCT6798D, driver nct6775; ADC de 8 mV, atualiza ~1x/s, não vê transiente
de milissegundos). Mapeamento da Pro WS X570-ACE ESTIMADO pelos divisores usuais da ASUS: +12V = in6 x 12 (passo de
96 mV), +5V = in1 x 5, +3.3V = in3. Cada linha mostra o valor do segundo; "mín" é o menor desde o início.
Roda como root: o energy_uj do RAPL é restrito. Não altera nada no hardware."""
import argparse
import ctypes as C
import datetime
import time
from pathlib import Path

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
TRILHOS = (("12V", "in6", 12.0), ("5V", "in1", 5.0), ("3.3V", "in3", 1.0))


def superio():
    for h in Path("/sys/class/hwmon").iterdir():
        if (h / "name").read_text().strip().startswith("nct67"):
            return h
    return None


def tensoes(h):
    return {nome: int((h / f"{inp}_input").read_text()) * k / 1000 for nome, inp, k in TRILHOS} if h else {}


class _Val(C.Union):
    _fields_ = [("d", C.c_double), ("ui", C.c_uint), ("ul", C.c_ulong), ("ull", C.c_ulonglong), ("sll", C.c_longlong)]


class _FV(C.Structure):
    _fields_ = [("fieldId", C.c_uint), ("scopeId", C.c_uint), ("timestamp", C.c_longlong),
                ("latencyUsec", C.c_longlong), ("valueType", C.c_int), ("nvmlReturn", C.c_int), ("value", _Val)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resto-w", type=float, default=60.0)
    p.add_argument("--fonte-w", type=float, default=0.0)
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    nv = C.CDLL("libnvidia-ml.so.1")
    nv.nvmlInit_v2()
    n = C.c_uint()
    nv.nvmlDeviceGetCount_v2(C.byref(n))
    hs = []
    for i in range(n.value):
        h = C.c_void_p()
        nv.nvmlDeviceGetHandleByIndex_v2(i, C.byref(h))
        hs.append(h)
    rmax = int(open(RAPL_MAX).read())
    e0, t0 = int(open(RAPL).read()), time.monotonic()
    out = open(a.csv, "a", buffering=1) if a.csv else None
    if out:
        out.write("timestamp," + ",".join(f"gpu{i}_w" for i in range(len(hs))) + ",gpus_w,cpu_w,v12,v5,v33\n")
    pico = 0.0
    sio = superio()
    vmin = {}
    v = tensoes(sio)
    gpus_hdr = " ".join(f"{'G' + str(i):>9}" for i in range(len(hs)))
    trilhos_hdr = "   12V   5V  3.3V (mín 12V/5V/3.3V)" if sio else ""
    print(f"hora     {gpus_hdr} |  ΣGPU méd/pico |  CPU | medido | +resto(est) | tomada(est){'  fonte' if a.fonte_w else ''} | pico medido |{trilhos_hdr}")
    while True:
        seg = [[] for _ in hs]
        somas, cpus = [], []
        fim = time.monotonic() + 1.0
        while time.monotonic() < fim:
            ws = []
            for i, h in enumerate(hs):
                fv = (_FV * 1)()
                fv[0].fieldId = 186
                nv.nvmlDeviceGetFieldValues(h, 1, fv)
                w = fv[0].value.ui / 1000 if fv[0].nvmlReturn == 0 else 0.0
                seg[i].append(w)
                ws.append(w)
            e1, t1 = int(open(RAPL).read()), time.monotonic()
            cpu = ((e1 - e0) % rmax) / 1e6 / (t1 - t0) if t1 > t0 else 0.0
            e0, t0 = e1, t1
            somas.append(sum(ws))
            cpus.append(cpu)
            if out:
                out.write(datetime.datetime.now().isoformat(timespec="milliseconds") + ","
                          + ",".join(f"{w:.1f}" for w in ws) + f",{sum(ws):.1f},{cpu:.1f},"
                          + ",".join(f"{v.get(k, 0):.3f}" for k in ("12V", "5V", "3.3V")) + "\n")
            time.sleep(0.1)
        cpu = sum(cpus) / len(cpus)
        med = sum(somas) / len(somas)
        medido = med + cpu
        pico = max(pico, max(somas) + max(cpus))
        tomada = (medido + a.resto_w) / 0.90
        gpus = " ".join(f"{sum(s) / len(s):4.0f}/{max(s):3.0f}" for s in seg)
        fonte = f" {100 * (medido + a.resto_w) / a.fonte_w:5.1f}%" if a.fonte_w else ""
        v = tensoes(sio)
        for k, x in v.items():
            vmin[k] = min(vmin.get(k, x), x)
        trilhos = (f" {v['12V']:5.2f} {v['5V']:4.2f} {v['3.3V']:5.3f} ({vmin['12V']:.2f}/{vmin['5V']:.2f}/{vmin['3.3V']:.3f})"
                   if v else "")
        print(f"{time.strftime('%H:%M:%S')} {gpus} | {med:5.0f} / {max(somas):5.0f} | {cpu:4.0f} | {medido:6.0f} | "
              f"{medido + a.resto_w:11.0f} | {tomada:11.0f}{fonte} | {pico:11.0f} |{trilhos}", flush=True)


if __name__ == "__main__":
    main()
