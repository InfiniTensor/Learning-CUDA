$MUSA_ROOT/bin/mcc -O3 -std=c++17 \
-I$MUSA_ROOT/include \
-L$MUSA_ROOT/lib \
-L/usr/lib/gcc/x86_64-linux-gnu/11 \
-lmusart -lstdc++ \
nf4_dequant_musa.mu -o nf4_musa