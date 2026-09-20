#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "==== [1/4] build ===="
bash build.sh
echo "==== [2/4] smoke 2body (100 steps, params_fast) ===="
./build/nbody data/particles_2body.txt data/params_fast.txt outputs/t2.bin outputs/p.log --check-energy
echo "==== [3/4] smoke 3body (100 steps) ===="
./build/nbody data/particles_3body.txt data/params_fast.txt outputs/t3.bin outputs/p.log --no-cpu
echo "==== [4/4] smoke 4096 (params_fast, 100 steps) ===="
./build/nbody data/particles_4096.txt data/params_fast.txt outputs/t4096.bin outputs/p.log --no-cpu
echo "==== all smoke runs ok ===="
