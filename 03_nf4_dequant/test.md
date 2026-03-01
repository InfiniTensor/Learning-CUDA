# benchmark_nf4.py
用来测试多组数据的加速比
# profile_nf4_only.py
只用来测试一组，利用ncu来进行分析
# nf4_cuda.cu
进行测试
编译命令：nvcc --shared -o libnf4_cuda.so nf4_cuda.cu -Xcompiler -fPIC
提供给python调用