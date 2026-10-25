"""vf_offset.py get | set IDX=MHZ ... | mem IDX=MHZ ... | reset — offsets V/F de GPC e de memória por GPU (NVML documentado).

Roda no host como root (sudo python3 -I vf_offset.py ...). Offset positivo desloca a curva inteira para cima:
com o lock de clock (-lgc), a GPU chega ao mesmo clock com menos voltagem (undervolt). Não persiste entre boots.
`mem` ajusta o offset de clock da memória (nvmlDeviceSetMemClkVfOffset); `reset` zera os dois.
O NVML usa índice de enumeração; o mapa índice → UUID sai no `get`."""
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
                raise SystemExit(f"GPU{i}: offset {mhz} não é múltiplo de 15 MHz (o driver arredonda para baixo)")
            check(nv.nvmlDeviceSetGpcClkVfOffset(handle(i), mhz), f"set {i}={mhz}")
    for i in range(n.value):
        print(f"GPU{i} {uuid(i)} offset={get(i)} MHz mem={get_mem(i)} MHz")
    nv.nvmlShutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
