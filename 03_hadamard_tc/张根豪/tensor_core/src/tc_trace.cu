// Diagnostic-only translation unit: reuse the actual production transform.
// Build this INSTEAD OF hadamard_tc.cu, not in addition to it.
#include "hadamard_tc.cu"

namespace {
template <typename T, int D>
__global__ void trace_kernel(const T* input, float* intermediate, float* before_cast, int groups) {
  extern __shared__ __align__(32) float memory[];
  constexpr int cf = kLdmOperand * kTileDim * sizeof(T) / sizeof(float);
  T* h = reinterpret_cast<T*>(memory);
  T* a = reinterpret_cast<T*>(memory + cf);
  build_constant_matrices<T,D>(h,a);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int group = blockIdx.x * kWarpsPerBlock + warp;
  if (group >= groups) return;
  float* scratch = memory + 2*cf + warp*warp_scratch_floats<T,true>();
  T* hi = reinterpret_cast<T*>(scratch + kLdmAcc*kTileDim);
  T* lo = hi + kLdmOperand*kTileDim;
  constexpr int tiles = TcShape<D>::kTilesPerToken;
  const size_t base = static_cast<size_t>(group)*TcShape<D>::kElemsPerWarp;
  float result[tiles*8];
  transform_tiles<T,D,false,true>(input,base,lane,1.0f,scratch,hi,lo,h,a,result);
  for (int t=0;t<tiles;++t) for (int i=0;i<8;++i)
    before_cast[base+t*256+lane*8+i]=result[t*8+i];
  wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> xf;
  wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> hf;
  wmma::fragment<wmma::accumulator,16,16,16,float> pf;
  wmma::load_matrix_sync(hf,h,kLdmOperand);
  for (int t=0;t<tiles;++t) {
    wmma::load_matrix_sync(xf,input+base+t*256,16);
    wmma::fill_fragment(pf,0.0f);
    wmma::mma_sync(pf,xf,hf,pf);
    wmma::store_matrix_sync(intermediate+base+t*256,pf,16,wmma::mem_row_major);
  }
}
template <typename T,int D>
int trace_dim(const void* x,float* p,float* y,int tokens,cudaStream_t stream) {
  constexpr int per = TcShape<D>::kTokensPerTile;
  if (tokens%per) return -10;
  const int groups = tokens/per;
  constexpr int cf = kLdmOperand*kTileDim*sizeof(T)/sizeof(float);
  trace_kernel<T,D><<<(groups+kWarpsPerBlock-1)/kWarpsPerBlock,kWarpsPerBlock*32,
      (2*cf+kWarpsPerBlock*warp_scratch_floats<T,true>())*sizeof(float),stream>>>
      (static_cast<const T*>(x),p,y,groups);
  return static_cast<int>(cudaGetLastError());
}
template <typename T>
int trace_typed(const void* x,float* p,float* y,int tokens,int dim,cudaStream_t stream) {
  switch(dim) {
    case 64:return trace_dim<T,64>(x,p,y,tokens,stream);
    case 128:return trace_dim<T,128>(x,p,y,tokens,stream);
    case 256:return trace_dim<T,256>(x,p,y,tokens,stream);
    case 512:return trace_dim<T,512>(x,p,y,tokens,stream);
    case 1024:return trace_dim<T,1024>(x,p,y,tokens,stream);
    default:return -10;
  }
}
}
extern "C" int hadamard_tc_trace(const void* x,float* p,float* y,int tokens,int dim,int dtype,cudaStream_t stream) {
  if (!x || !p || !y || tokens<=0 || (dtype!=0 && dtype!=1)) return -10;
  return dtype==0 ? trace_typed<__half>(x,p,y,tokens,dim,stream)
                  : trace_typed<__nv_bfloat16>(x,p,y,tokens,dim,stream);
}
