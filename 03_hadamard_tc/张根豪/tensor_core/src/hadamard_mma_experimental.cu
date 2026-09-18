// Experimental SM80+ PTX path, isolated from production dispatch.
// Layout follows PTX ISA "Matrix Fragments for mma.m16n8k16 with floating point type":
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
// Two n=8 halves cover each 16x16 tile. C->B conversion uses explicit warp
// shuffles, not undocumented WMMA fragment indices. Zero shared memory.
#include "hadamard.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Numerical experiments only: 0 preserves the original split algorithm;
// 1 accumulates high/low MMA separately; 2 adds a third native residual;
// 3 adds a conservative output-interval guard and full-tile warp fallback.
// None implies strict equivalence to the FP32 butterfly reference.
#ifndef HADAMARD_MMA_PRECISION
#define HADAMARD_MMA_PRECISION 0
#endif
static_assert(HADAMARD_MMA_PRECISION >= 0 && HADAMARD_MMA_PRECISION <= 3);

namespace {
template <bool BF> __device__ unsigned short bits(float v) {
  if constexpr (BF) return __bfloat16_as_ushort(__float2bfloat16_rn(v));
  else return __half_as_ushort(__float2half_rn(v));
}
template <bool BF> __device__ float value(unsigned short v) {
  if constexpr (BF) return __bfloat162float(__ushort_as_bfloat16(v));
  else return __half2float(__ushort_as_half(v));
}
template <bool BF> __device__ unsigned pack(float a,float b) {
  return static_cast<unsigned>(bits<BF>(a)) | (static_cast<unsigned>(bits<BF>(b))<<16);
}
template <bool BF>
__device__ __forceinline__ void mma(const unsigned (&a)[4],const unsigned (&b)[2],float (&d)[4]) {
  if constexpr (BF) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
}
__device__ float sign(int r,int c) { return (__popc(r&c)&1) ? -1.f : 1.f; }

// Same FP32 butterfly order as the reference-compatible warp path. A guarded
// tile owns exactly 256 elements; all lanes recompute all its tokens together.
template <bool BF,int D>
__device__ void strict_tile(const unsigned short* input,unsigned short* output,float scale,int lane) {
  constexpr int E=D/32;
#pragma unroll
  for(int token=0;token<256/D;++token) {
    float v[E];
#pragma unroll
    for(int j=0;j<E;++j) v[j]=value<BF>(input[token*D+lane*E+j]);
#pragma unroll
    for(int s=1;s<E;s*=2) {
#pragma unroll
      for(int j=0;j<E;++j) if(!(j&s)) {
        const float a=v[j],b=v[j|s];v[j]=a+b;v[j|s]=a-b;
      }
    }
#pragma unroll
    for(int s=1;s<32;s*=2) {
#pragma unroll
      for(int j=0;j<E;++j) {
        const float other=__shfl_xor_sync(0xffffffffu,v[j],s);
        v[j]=(lane&s) ? other-v[j] : v[j]+other;
      }
    }
#pragma unroll
    for(int j=0;j<E;j+=2)
      *reinterpret_cast<unsigned*>(output+token*D+lane*E+j)=pack<BF>(v[j]*scale,v[j+1]*scale);
  }
}

template <bool BF,int D,bool SPLIT>
__global__ void register_mma_kernel(const unsigned short* input,unsigned short* output,int groups,float scale) {
  const int lane=threadIdx.x&31, group=lane>>2, tid=lane&3;
  const int tile=blockIdx.x*4+(threadIdx.x>>5);
  if (tile>=groups) return;  // whole warp uniform
  const size_t base=static_cast<size_t>(tile)*256;
  unsigned x[4],a[4];
  constexpr int m=D/16;
#pragma unroll
  for (int reg=0;reg<4;++reg) {
    const int row=group+(reg%2)*8, col=tid*2+(reg/2)*8;
    x[reg]=*reinterpret_cast<const unsigned*>(input+base+row*16+col);
    const float v0=row/m==col/m ? sign(row%m,col%m) : 0.f;
    const float v1=row/m==(col+1)/m ? sign(row%m,(col+1)%m) : 0.f;
    a[reg]=pack<BF>(v0,v1);
  }
  float error_bound=0;
  if constexpr (SPLIT && HADAMARD_MMA_PRECISION == 3) {
    float maximum=0;
#pragma unroll
    for(int r=0;r<4;++r) {
      maximum=fmaxf(maximum,fabsf(value<BF>(x[r]&65535)));
      maximum=fmaxf(maximum,fabsf(value<BF>(x[r]>>16)));
    }
#pragma unroll
    for(int s=16;s;s/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,s));
    // Deliberately conservative envelope for two length<=16 FP32 MMAs,
    // three native residual terms, FP32 merges/scale and reference butterflies.
    // eps is 2^-23 (twice round-to-nearest unit roundoff); upward rounding.
    // The absolute floor covers subnormal/FTZ effects, far below PDF tolerances.
    error_bound=__fadd_ru(__fmul_ru(__fmul_ru(maximum,128.f*0x1p-23f*D),scale),1.e-30f);
  }
  float p[2][4]={{0,0,0,0},{0,0,0,0}};
#pragma unroll
  for (int half=0;half<2;++half) {
    unsigned h[2];
#pragma unroll
    for (int reg=0;reg<2;++reg) {
      const int row=2*tid+reg*8,col=group+half*8;
      h[reg]=pack<BF>(sign(row,col),sign(row+1,col));
    }
    mma<BF>(x,h,p[half]);
  }
#pragma unroll
  for (int half=0;half<2;++half) {
    unsigned hi[2],lo[2],third[2];
#pragma unroll
    for (int reg=0;reg<2;++reg) {
      float vals[2];
#pragma unroll
      for (int j=0;j<2;++j) {
        // B[row=2*tid+j+8*reg,col=group+8*half] comes from
        // C in source lane 4*(row%8)+(col%8)/2, element 2*(row/8)+(col%2).
        const int source=4*(2*tid+j)+group/2;
        const float even=__shfl_sync(0xffffffffu,p[half][reg*2],source);
        const float odd=__shfl_sync(0xffffffffu,p[half][reg*2+1],source);
        vals[j]=(group&1) ? odd : even;
      }
      hi[reg]=pack<BF>(vals[0],vals[1]);
      if constexpr (SPLIT) lo[reg]=pack<BF>(vals[0]-value<BF>(hi[reg]&65535),vals[1]-value<BF>(hi[reg]>>16));
      if constexpr (SPLIT && HADAMARD_MMA_PRECISION >= 2)
        third[reg]=pack<BF>((vals[0]-value<BF>(hi[reg]&65535))-value<BF>(lo[reg]&65535),
                            (vals[1]-value<BF>(hi[reg]>>16))-value<BF>(lo[reg]>>16));
    }
    float y[4]={0,0,0,0};
    mma<BF>(a,hi,y);
    if constexpr (SPLIT) {
      if constexpr (HADAMARD_MMA_PRECISION == 0) mma<BF>(a,lo,y);
      else {
        float residual[4]={0,0,0,0};
        mma<BF>(a,lo,residual);
        if constexpr (HADAMARD_MMA_PRECISION >= 2) {
          float tail[4]={0,0,0,0};
          mma<BF>(a,third,tail);
#pragma unroll
          for (int j=0;j<4;++j) residual[j]+=tail[j];
        }
#pragma unroll
        for (int j=0;j<4;++j) y[j]+=residual[j];
      }
    }
    if constexpr (SPLIT && HADAMARD_MMA_PRECISION == 3) {
      bool unsafe=false;
#pragma unroll
      for(int j=0;j<4;++j) {
        const float center=y[j]*scale;
        const float native=value<BF>(bits<BF>(center));
        const float lower=value<BF>(bits<BF>(__fsub_rd(center,error_bound)));
        const float upper=value<BF>(bits<BF>(__fadd_ru(center,error_bound)));
        const float difference=fmaxf(fabsf(native-lower),fabsf(native-upper));
        unsafe |= !isfinite(center) || !isfinite(error_bound) || !isfinite(native) ||
                  !isfinite(lower) || !isfinite(upper) || !(difference < (BF ? .05f : .01f));
      }
      if(__any_sync(0xffffffffu,unsafe)) {
        // Half 0 might already have stored: order those writes before a
        // different lane overwrites the full tile in the fallback layout.
        __syncwarp();
        strict_tile<BF,D>(input+base,output+base,scale,lane);
        return;
      }
    }
#pragma unroll
    for (int rowhalf=0;rowhalf<2;++rowhalf) {
      const int row=group+8*rowhalf,col=2*tid+8*half;
      *reinterpret_cast<unsigned*>(output+base+row*16+col)=pack<BF>(y[2*rowhalf]*scale,y[2*rowhalf+1]*scale);
    }
  }
}
template <bool BF,int D>
void launch(const void* x,void* y,int groups,bool split,float scale,cudaStream_t stream) {
  if (split) register_mma_kernel<BF,D,true><<<(groups+3)/4,128,0,stream>>>(static_cast<const unsigned short*>(x),static_cast<unsigned short*>(y),groups,scale);
  else register_mma_kernel<BF,D,false><<<(groups+3)/4,128,0,stream>>>(static_cast<const unsigned short*>(x),static_cast<unsigned short*>(y),groups,scale);
}
}
extern "C" int hadamard_mma_run(const void* x,void* y,int tokens,int dim,int dtype,int norm,int split,cudaStream_t stream) {
  if (tokens<=0 || (dim!=64 && dim!=128 && dim!=256) || dtype<0 || dtype>1) return -10;
  const int per=256/dim,groups=tokens/per,complete=groups*per;
  const float scale=norm ? static_cast<float>(1.0/std::sqrt(static_cast<double>(dim))) : 1.f;
  if (groups) {
#define DISPATCH(D) if (dtype) launch<true,D>(x,y,groups,split,scale,stream); else launch<false,D>(x,y,groups,split,scale,stream)
    switch(dim) { case 64: { DISPATCH(64); break; } case 128: { DISPATCH(128); break; } case 256: { DISPATCH(256); break; } }
#undef DISPATCH
    const auto err=cudaGetLastError(); if (err!=cudaSuccess) return static_cast<int>(err);
  }
  if (complete<tokens) return launch_hadamard_warp(static_cast<const unsigned short*>(x)+static_cast<size_t>(complete)*dim,
      static_cast<unsigned short*>(y)+static_cast<size_t>(complete)*dim,1,1,tokens-complete,dim,
      dtype ? DataType::BF16 : DataType::FP16,norm,stream);
  return 0;
}
