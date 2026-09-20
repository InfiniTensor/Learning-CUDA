#!/usr/bin/env python3
import struct, pathlib, sys
for p in sys.argv[1:]:
    data = pathlib.Path(p).read_bytes()
    N, R = struct.unpack_from("<ii", data, 0)
    print(f"{p}: N={N} R={R} bytes={len(data)} expected={8+R*N*3*4} OK={len(data)==8+R*N*3*4}")
