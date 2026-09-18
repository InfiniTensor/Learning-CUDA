// Standalone NCU workload: exactly four selected launches, capture the fourth.
#include "hadamard_tc.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cmath>
extern "C" int hadamard_mma_run(const void*,void*,int,int,int,int,int,cudaStream_t);
#define CHECK(x) do { if ((x)!=0) { std::fprintf(stderr,"failed: %s\n",#x); return 1; } } while(0)
int main(int argc,char** argv) {
  const int d=argc>1 ? std::atoi(argv[1]) : 128;
  const int mode=argc>2 ? std::atoi(argv[2]) : 0;
  const int n=16384;
  if ((d!=64 && d!=128 && d!=256) || mode<0 || mode>4) return 2;
  std::mt19937 rng(20260910); std::normal_distribution<float> dist;
  std::vector<__half> input(n*d),output(n*d);
  for(auto& v:input) v=__float2half_rn(dist(rng));
  void *x,*y; CHECK(cudaMalloc(&x,input.size()*2)); CHECK(cudaMalloc(&y,input.size()*2));
  CHECK(cudaMemcpy(x,input.data(),input.size()*2,cudaMemcpyHostToDevice));
  for(int i=0;i<4;++i) {
    if(mode==0) { CHECK(launch_hadamard_warp(x,y,1,1,n,d,DataType::FP16,true,nullptr)); }
    else if(mode<=2) { CHECK(launch_hadamard_tc(x,y,1,1,n,d,DataType::FP16,true,mode==1?TcMode::Fast:TcMode::Split,nullptr)); }
    else { CHECK(hadamard_mma_run(x,y,n,d,0,1,mode==4,nullptr)); }
  }
  CHECK(cudaMemcpy(output.data(),y,output.size()*2,cudaMemcpyDeviceToHost));
  for(auto v:output) if(!std::isfinite(__half2float(v))) return 3;
  CHECK(cudaFree(x)); CHECK(cudaFree(y));
  std::printf("dim=%d mode=%d tokens=%d launches=4 finite=true\n",d,mode,n);
}
