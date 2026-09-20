#include "kernel_operator.h"
using namespace AscendC;
#ifndef LP_ASCEND_CUBE_ROWS
#define LP_ASCEND_CUBE_ROWS 64
#endif
constexpr int TILE_ROWS = LP_ASCEND_CUBE_ROWS;
static_assert(TILE_ROWS >= 16 && TILE_ROWS <= 128 && TILE_ROWS % 16 == 0);

// Each Cube core computes TILE_ROWS x 16 outputs. ND->NZ copy and Fixpipe handle
// partial final row tiles; accumulation stays FP32 until the AIV epilogue.
// One native Mmad computes H16; FP32 vector butterflies complete larger heads.
template <class T>
__aicore__ inline void cube_product(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                    uint64_t rows, int d) {
  TPipe pipe;
  TBuf<TPosition::A1> abuf;
  TBuf<TPosition::B1> bbuf;
  TBuf<TPosition::A2> a0buf;
  TBuf<TPosition::B2> b0buf;
  TBuf<TPosition::CO1> cbuf;
  pipe.InitBuffer(abuf, TILE_ROWS * 32);
  pipe.InitBuffer(bbuf, 512);
  pipe.InitBuffer(a0buf, TILE_ROWS * 32);
  pipe.InitBuffer(b0buf, 512);
  pipe.InitBuffer(cbuf, TILE_ROWS * 64);
  auto a1 = abuf.Get<T>(), b1 = bbuf.Get<T>(), a0 = a0buf.Get<T>(), b0 = b0buf.Get<T>();
  auto c0 = cbuf.Get<float>();
  GlobalTensor<T> a, b;
  GlobalTensor<float> c;
  a.SetGlobalBuffer((__gm__ T *)input);
  b.SetGlobalBuffer((__gm__ T *)weight);
  c.SetGlobalBuffer((__gm__ float *)output);
  // The H16 coefficient tile is invariant across all output tiles on a core.
  Nd2NzParams bp;
  bp.ndNum = 1;
  bp.nValue = 16;
  bp.dValue = 16;
  bp.srcDValue = 16;
  bp.dstNzC0Stride = 16;
  bp.dstNzNStride = 1;
  DataCopy(b1, b, bp);
  PipeBarrier<PIPE_ALL>();
  LoadData2DParams lb;
  lb.repeatTimes = 1;
  lb.srcStride = 1;
  lb.ifTranspose = true;
  LoadData(b0, b1, lb);
  PipeBarrier<PIPE_ALL>();
  uint64_t tiles = (rows + TILE_ROWS - 1) / TILE_ROWS;
  for (uint64_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum()) {
    uint64_t r = tile * TILE_ROWS;
    int valid = rows - r < TILE_ROWS ? int(rows - r) : TILE_ROWS;
    int padded = (valid + 15) / 16 * 16;
    Nd2NzParams ap;
    ap.ndNum = 1;
    ap.nValue = valid;
    ap.dValue = 16;
    ap.srcDValue = 16;
    ap.dstNzC0Stride = padded;
    ap.dstNzNStride = 1;
    DataCopy(a1, a[r * 16], ap);
    PipeBarrier<PIPE_ALL>();
    LoadData2DParams la;
    la.repeatTimes = padded / 16;
    la.srcStride = 1;
    la.ifTranspose = false;
    LoadData(a0, a1, la);
    PipeBarrier<PIPE_ALL>();
    MmadParams mm;
    mm.m = padded;
    mm.n = 16;
    mm.k = 16;
    mm.isBias = false;
    Mmad(c0, a0, b0, mm);
    PipeBarrier<PIPE_ALL>();
    FixpipeParamsV220 fix;
    fix.nSize = 16;
    fix.mSize = valid;
    fix.srcStride = padded;
    fix.dstStride = 16;
    Fixpipe(c[r * 16], c0, fix);
    PipeBarrier<PIPE_ALL>();
  }
}
extern "C" __global__ __aicore__ void ascend_cube(GM_ADDR input, GM_ADDR weight,
                                                  GM_ADDR output, uint64_t rows, int d,
                                                  int dtype) {
  if (dtype == 1)
    cube_product<half>(input, weight, output, rows, d);
  else
    cube_product<bfloat16_t>(input, weight, output, rows, d);
}
#ifndef __CCE_AICORE__
extern "C" void launch_ascend_cube(void *stream, void *input, void *weight,
                                   void *output, uint64_t rows, int d, int dtype,
                                   unsigned cores) {
  ascend_cube<<<cores, nullptr, stream>>>((uint8_t *)input, (uint8_t *)weight,
                                          (uint8_t *)output, rows, d, dtype);
}
#endif
