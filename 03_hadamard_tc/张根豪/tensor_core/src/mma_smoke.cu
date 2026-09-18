#include "hadamard_tc.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstdio>
extern "C" int hadamard_mma_run(const void*,void*,int,int,int,int,int,cudaStream_t);
#define CHECK(x) do { if ((x)!=0) { std::fprintf(stderr,"failed: %s\n",#x); return 1; } } while(0)
int main(int argc,char**) {
  const bool profile=argc>1;
  const std::vector<int> dims=profile ? std::vector<int>{128} : std::vector<int>{64,128,256};
  const std::vector<int> sizes=profile ? std::vector<int>{16384} : std::vector<int>{1,5,128};
  for (int dtype=0;dtype<(profile?1:2);++dtype) for (int dim : dims) for (int tokens : sizes) {
    std::vector<unsigned short> input(tokens*dim),a(input.size()),b(input.size());
    // Exact representable impulse/alternating finite values, no tolerance bypass.
    for (size_t i=0;i<input.size();++i) {
      float value = ((i%7)-3.0f)*.125f;
      input[i]=dtype ? __bfloat16_as_ushort(__float2bfloat16_rn(value)) : __half_as_ushort(__float2half_rn(value));
    }
    void *x,*y,*z;
    CHECK(cudaMalloc(&x,input.size()*2)); CHECK(cudaMalloc(&y,input.size()*2)); CHECK(cudaMalloc(&z,input.size()*2));
    CHECK(cudaMemcpy(x,input.data(),input.size()*2,cudaMemcpyHostToDevice));
    for (int split=0;split<(profile?1:2);++split) for (int norm=profile?1:0;norm<2;++norm) {
      CHECK(hadamard_mma_run(x,y,tokens,dim,dtype,norm,split,nullptr));
      CHECK(launch_hadamard_tc(x,z,1,1,tokens,dim,dtype ? DataType::BF16 : DataType::FP16,norm,split ? TcMode::Split : TcMode::Fast,nullptr));
      CHECK(cudaMemcpy(a.data(),y,input.size()*2,cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(b.data(),z,input.size()*2,cudaMemcpyDeviceToHost));
      if (a!=b) { std::fprintf(stderr,"MMA/WMMA mismatch d=%d tokens=%d dtype=%d split=%d norm=%d\n",dim,tokens,dtype,split,norm); return 1; }
    }
    CHECK(cudaFree(x)); CHECK(cudaFree(y)); CHECK(cudaFree(z));
  }
  std::puts(profile ? "profile pair exact-pattern check passed" : "72 MMA/WMMA exact-pattern comparisons passed");
}
