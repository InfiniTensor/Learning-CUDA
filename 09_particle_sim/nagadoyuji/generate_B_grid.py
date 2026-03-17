#!/usr/bin/env python3
"""Generate a binary B-field grid file for 3D grid mode.
Format: 3 int32 (nx,ny,nz), 3 float32 (ox,oy,oz), 3 float32 (dx,dy,dz),
        then nx*ny*nz*3 float32 (Bx,By,Bz per cell, order i,j,k)."""
import struct
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="data/field_grid.bin")
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--ny", type=int, default=4)
    ap.add_argument("--nz", type=int, default=4)
    ap.add_argument("--ox", type=float, default=-0.01)
    ap.add_argument("--oy", type=float, default=-0.01)
    ap.add_argument("--oz", type=float, default=-0.01)
    ap.add_argument("--dx", type=float, default=0.01)
    ap.add_argument("--dy", type=float, default=0.01)
    ap.add_argument("--dz", type=float, default=0.01)
    ap.add_argument("--Bz", type=float, default=1.0)
    args = ap.parse_args()
    nx, ny, nz = args.nx, args.ny, args.nz
    ox, oy, oz = args.ox, args.oy, args.oz
    dx, dy, dz = args.dx, args.dy, args.dz
    with open(args.output, "wb") as f:
        f.write(struct.pack("iii", nx, ny, nz))
        f.write(struct.pack("fff", ox, oy, oz))
        f.write(struct.pack("fff", dx, dy, dz))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(struct.pack("fff", 0.0, 0.0, args.Bz))
    print(f"Wrote grid {nx}x{ny}x{nz} to {args.output}")

if __name__ == "__main__":
    main()
