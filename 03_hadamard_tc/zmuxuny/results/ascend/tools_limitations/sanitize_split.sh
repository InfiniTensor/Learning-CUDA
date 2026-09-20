set -eu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/camp/probes
inc=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw
flags="-O2 -g -std=c++17 -xcce --cce-aicore-arch=dav-c220-vec -I$inc -I$inc/interface -I$inc/impl -I/usr/local/Ascend/ascend-toolkit/latest/include"
bisheng $flags --cce-aicore-only --cce-enable-sanitizer -c probe_san.cpp -o probe_device.o
bisheng $flags --cce-host-only --cce-aicore-bin=probe_device.o -fPIC -shared probe_san.cpp -o probe_san_split.so -L/usr/local/Ascend/ascend-toolkit/latest/lib64 -lascendcl -lruntime
