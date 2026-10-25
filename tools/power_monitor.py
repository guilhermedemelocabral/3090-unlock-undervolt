"""power_monitor.py [--rest-w W] [--psu-w W] [--csv FILE] — system power draw, one line per second.

Samples every 100 ms:
- each GPU: instantaneous board power (NVML field 186);
- CPU: package energy (RAPL).
Each line prints, for that second, every GPU's mean/peak, the GPU sum (mean and peak of the instantaneous sum), the
CPU, the measured total and an estimate. The estimate adds --rest-w (motherboard, RAM, disks and fans; NOT measured,
default 60 W) and divides by 0.90 PSU efficiency for the wall figure. With --psu-w it shows the load as % of the PSU.
"peak" is the highest measured total (GPUs + CPU, peak) since start.
PSU rail voltages come from the motherboard Super I/O (NCT6798D, nct6775 driver; 8 mV ADC, updates ~1x/s, so it cannot
see millisecond transients). The mapping used here is an ESTIMATE from the usual ASUS dividers (checked on a Pro WS
X570-ACE: the 12 V reading rose from 11.90 to 12.00 V at idle): +12V = in6 x 12 (96 mV steps), +5V = in1 x 5,
+3.3V = in3. Check your board. "min" is the lowest value since start.
Run as root: RAPL energy_uj is restricted. It does not change anything on the hardware."""
import argparse
import ctypes as C
import datetime
import time
from pathlib import Path

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
RAILS = (("12V", "in6", 12.0), ("5V", "in1", 5.0), ("3.3V", "in3", 1.0))


def superio():
    for h in Path("/sys/class/hwmon").iterdir():
        if (h / "name").read_text().strip().startswith("nct67"):
            return h
    return None


def voltages(h):
    return {name: int((h / f"{inp}_input").read_text()) * k / 1000 for name, inp, k in RAILS} if h else {}


class _Val(C.Union):
    _fields_ = [("d", C.c_double), ("ui", C.c_uint), ("ul", C.c_ulong), ("ull", C.c_ulonglong), ("sll", C.c_longlong)]


class _FV(C.Structure):
    _fields_ = [("fieldId", C.c_uint), ("scopeId", C.c_uint), ("timestamp", C.c_longlong),
                ("latencyUsec", C.c_longlong), ("valueType", C.c_int), ("nvmlReturn", C.c_int), ("value", _Val)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rest-w", type=float, default=60.0)
    p.add_argument("--psu-w", type=float, default=0.0)
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
    peak = 0.0
    sio = superio()
    vmin = {}
    v = voltages(sio)
    gpus_hdr = " ".join(f"{'G' + str(i):>9}" for i in range(len(hs)))
    rails_hdr = "   12V   5V  3.3V (min 12V/5V/3.3V)" if sio else ""
    print(f"time     {gpus_hdr} |  ΣGPU mean/peak |  CPU | measured | +rest(est) | wall(est){'   PSU' if a.psu_w else ''} | peak |{rails_hdr}")
    while True:
        sec = [[] for _ in hs]
        sums, cpus = [], []
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            ws = []
            for i, h in enumerate(hs):
                fv = (_FV * 1)()
                fv[0].fieldId = 186
                nv.nvmlDeviceGetFieldValues(h, 1, fv)
                w = fv[0].value.ui / 1000 if fv[0].nvmlReturn == 0 else 0.0
                sec[i].append(w)
                ws.append(w)
            e1, t1 = int(open(RAPL).read()), time.monotonic()
            cpu = ((e1 - e0) % rmax) / 1e6 / (t1 - t0) if t1 > t0 else 0.0
            e0, t0 = e1, t1
            sums.append(sum(ws))
            cpus.append(cpu)
            if out:
                out.write(datetime.datetime.now().isoformat(timespec="milliseconds") + ","
                          + ",".join(f"{w:.1f}" for w in ws) + f",{sum(ws):.1f},{cpu:.1f},"
                          + ",".join(f"{v.get(k, 0):.3f}" for k in ("12V", "5V", "3.3V")) + "\n")
            time.sleep(0.1)
        cpu = sum(cpus) / len(cpus)
        mean = sum(sums) / len(sums)
        measured = mean + cpu
        peak = max(peak, max(sums) + max(cpus))
        wall = (measured + a.rest_w) / 0.90
        gpus = " ".join(f"{sum(s) / len(s):4.0f}/{max(s):3.0f}" for s in sec)
        psu = f" {100 * (measured + a.rest_w) / a.psu_w:5.1f}%" if a.psu_w else ""
        v = voltages(sio)
        for k, x in v.items():
            vmin[k] = min(vmin.get(k, x), x)
        rails = (f" {v['12V']:5.2f} {v['5V']:4.2f} {v['3.3V']:5.3f} ({vmin['12V']:.2f}/{vmin['5V']:.2f}/{vmin['3.3V']:.3f})"
                 if v else "")
        print(f"{time.strftime('%H:%M:%S')} {gpus} | {mean:5.0f} / {max(sums):5.0f} | {cpu:4.0f} | {measured:8.0f} | "
              f"{measured + a.rest_w:10.0f} | {wall:9.0f}{psu} | {peak:4.0f} |{rails}", flush=True)


if __name__ == "__main__":
    main()
