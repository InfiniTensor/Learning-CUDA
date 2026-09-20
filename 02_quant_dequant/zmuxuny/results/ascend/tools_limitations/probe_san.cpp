#include "kernel_operator.h"
extern "C" __global__ __aicore__ void probe(GM_ADDR x, GM_ADDR y) {
  AscendC::GlobalTensor<float> a,b;
  a.SetGlobalBuffer((__gm__ float*)x); b.SetGlobalBuffer((__gm__ float*)y);
  float f=a.GetValue(AscendC::GetBlockIdx());
  b.SetValue(AscendC::GetBlockIdx(), f / 3.0f);
}
#ifndef ASCENDC_CPU_DEBUG
extern "C" void launch_probe(unsigned n, void *stream, void *x, void *y) {
  probe<<<n, nullptr, stream>>>((uint8_t*)x,(uint8_t*)y);
}
#endif
