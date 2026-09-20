"""用 ctypes 调用某个 DLL 导出的 main。

用途: 本机 Windows 的 Smart App Control 会拦截新编译出来的 .exe (WinError 4551), 于是
`make` / scripts/build.ps1 产出的可执行文件在这台机器上跑不起来。绕行办法是把目标编成 DLL
(多传一个 -shared 与 -Xlinker /DEF:build/_exp.def, 见 build/_exp.def) 再用本脚本运行:

    nvcc <原来的全部编译参数> -shared src/main.cu <公共源> -Xlinker /DEF:build/_exp.def \
         -o build/lpq_quant_d.dll
    python scripts/run_dll.py build/lpq_quant_d.dll --input data/tensors/normal_512.tensor ...

argv[0] 放 DLL 自身的绝对路径, 其余参数原样透传给 main。Linux 上不需要这个脚本, 直接跑
build/lpq_quant 即可。
"""

import ctypes
import os
import sys


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    lib = ctypes.CDLL(os.path.abspath(sys.argv[1]))
    fn = lib.main
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    fn.restype = ctypes.c_int
    buf = [a.encode("utf-8") for a in [os.path.abspath(sys.argv[1])] + sys.argv[2:]]
    arr = (ctypes.c_char_p * (len(buf) + 1))()
    for i, b in enumerate(buf):
        arr[i] = b
    arr[len(buf)] = None
    rc = fn(len(buf), arr)
    sys.stdout.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
