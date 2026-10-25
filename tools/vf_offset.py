"""vf_offset.py get | set IDX=MHZ ... | mem IDX=MHZ ... | reset — per-GPU V/F offsets for GPC and memory (documented NVML).

Run on the host as root (sudo python3 -I vf_offset.py ...). A positive GPC offset shifts the whole V/F curve up:
together with a clock lock (-lgc) the GPU reaches the same clock at a lower voltage (undervolt). WITHOUT the lock the
same offset is an overclock. Offsets do not survive a reboot or a driver reload.
`mem` sets the memory clock offset (nvmlDeviceSetMemClkVfOffset). Its unit is 2x the MHz that nvidia-smi reports
(+500 gave +250 MHz on a 3090). `reset` sets both to zero.
NVML uses enumeration order; `get` prints the index -> UUID map."""
import ctypes
import sys

nv = ctypes.CDLL("libnvidia-ml.so.1")


def check(rc, what):
    if rc != 0:
        nv.nvmlErrorString.restype = ctypes.c_char_p
        raise SystemExit(f"{what}: NVML rc={rc} ({nv.nvmlErrorString(rc).decode()})")


def handle(i):
    h = ctypes.c_void_p()
    check(nv.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)), f"handle {i}")
    return h


def get(i):
    cur = ctypes.c_int()
    check(nv.nvmlDeviceGetGpcClkVfOffset(handle(i), ctypes.byref(cur)), f"get {i}")
    return cur.value


def get_mem(i):
    cur = ctypes.c_int()
    check(nv.nvmlDeviceGetMemClkVfOffset(handle(i), ctypes.byref(cur)), f"get mem {i}")
    return cur.value


def uuid(i):
    buf = ctypes.create_string_buffer(96)
    check(nv.nvmlDeviceGetUUID(handle(i), buf, 96), f"uuid {i}")
    return buf.value.decode()


def main(argv):
    check(nv.nvmlInit_v2(), "init")
    n = ctypes.c_uint()
    check(nv.nvmlDeviceGetCount_v2(ctypes.byref(n)), "count")
    cmd = argv[0] if argv else "get"
    if cmd == "reset":
        for i in range(n.value):
            check(nv.nvmlDeviceSetMemClkVfOffset(handle(i), 0), f"reset mem {i}")
        argv = ["set"] + [f"{i}=0" for i in range(n.value)]
        cmd = "set"
    if cmd == "mem":
        for kv in argv[1:]:
            i, mhz = (int(x) for x in kv.split("="))
            check(nv.nvmlDeviceSetMemClkVfOffset(handle(i), mhz), f"mem {i}={mhz}")
    if cmd == "set":
        for kv in argv[1:]:
            i, mhz = (int(x) for x in kv.split("="))
            if mhz % 15:
                raise SystemExit(f"GPU{i}: offset {mhz} is not a multiple of 15 MHz (the driver rounds down)")
            check(nv.nvmlDeviceSetGpcClkVfOffset(handle(i), mhz), f"set {i}={mhz}")
    for i in range(n.value):
        print(f"GPU{i} {uuid(i)} offset={get(i)} MHz mem={get_mem(i)} MHz")
    nv.nvmlShutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
