#define LP_ASCEND 1
#define LP_ASCEND_DEVICE 1
#include "../include/numeric.cuh"
using namespace AscendC;

// GM input is read through typed tensors; output DMA is byte-granular so odd
// rows and adjacent groups never race through scalar byte-store cache lines.
__aicore__ inline float input_value(GM_ADDR in, uint64_t i, int dtype) {
  if (dtype == lp::FP32) {
    GlobalTensor<float> x;
    x.SetGlobalBuffer((__gm__ float *)in);
    return x.GetValue(i);
  }
  if (dtype == lp::FP16) {
    GlobalTensor<half> x;
    x.SetGlobalBuffer((__gm__ half *)in);
    return float(x.GetValue(i));
  }
  GlobalTensor<uint16_t> x;
  x.SetGlobalBuffer((__gm__ uint16_t *)in);
  return lp::unbf16(x.GetValue(i));
}
__aicore__ inline void store_dma(GlobalTensor<uint8_t> dst, LocalTensor<uint8_t> src,
                                 uint32_t bytes) {
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopyPad(dst, src, DataCopyExtParams{1, bytes, 0, 0, 0});
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
__aicore__ inline void load_dma(LocalTensor<uint8_t> dst, GlobalTensor<uint8_t> src,
                                uint32_t bytes) {
  DataCopyPad(dst, src, DataCopyExtParams{1, bytes, 0, 0, 0},
              DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}
__aicore__ inline float local_value(LocalTensor<uint8_t> src, unsigned i, int dtype) {
  if (dtype == lp::FP32)
    return src.ReinterpretCast<float>().GetValue(i);
  if (dtype == lp::FP16)
    return float(src.ReinterpretCast<half>().GetValue(i));
  return lp::unbf16(src.ReinterpretCast<uint16_t>().GetValue(i));
}
extern "C" __global__ __aicore__ void ascend_amax(GM_ADDR in, GM_ADDR partial,
                                                  uint64_t n, int dtype) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf, vbuf, tmpbuf, reducebuf;
  pipe.InitBuffer(buf, 4096 * 4);
  pipe.InitBuffer(vbuf, 4096 * 4);
  pipe.InitBuffer(tmpbuf, 4096 * 4);
  pipe.InitBuffer(reducebuf, 256);
  auto raw = buf.Get<uint8_t>();
  auto work = vbuf.Get<float>();
  auto tmp = tmpbuf.Get<float>();
  auto reduced = reducebuf.Get<float>();
  GlobalTensor<uint8_t> x;
  x.SetGlobalBuffer(in);
  int width = dtype == lp::FP32 ? 4 : 2;
  float a = 0;
  for (uint64_t base = GetBlockIdx() * 4096; base < n; base += GetBlockNum() * 4096) {
    unsigned count = n - base < 4096 ? unsigned(n - base) : 4096;
    load_dma(raw, x[base * width], count * width);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    if (dtype == lp::FP32)
      Abs(work, raw.ReinterpretCast<float>(), count);
    else {
      if (dtype == lp::FP16)
        Cast(work, raw.ReinterpretCast<half>(), RoundMode::CAST_NONE, count);
      else
        Cast(work, raw.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      Abs(work, work, count);
    }
    PipeBarrier<PIPE_V>();
    ReduceMax(reduced, work, tmp, int(count), false);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float m = reduced.GetValue(0);
    // Vector math may flush an all-subnormal tile. Inspect its original bits
    // only in this rare case, preserving the software format's tiny scales.
    if (m == 0 && dtype == lp::FP32)
      for (unsigned j = 0; j < count; ++j)
        m = lp::max_f(m, lp::abs_f(local_value(raw, j, dtype)));
    a = lp::max_f(a, m);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
  }
  GlobalTensor<float> out;
  out.SetGlobalBuffer((__gm__ float *)partial);
  out.SetValue(GetBlockIdx() * 16, a);
  // Restore the ABI vector mask after the final partial reduction tile.
  ResetMask();
}
extern "C" __global__ __aicore__ void ascend_max_finish(GM_ADDR partial, GM_ADDR result,
                                                        int cores) {
  GlobalTensor<float> x, y;
  x.SetGlobalBuffer((__gm__ float *)partial);
  y.SetGlobalBuffer((__gm__ float *)result);
  float a = 0;
  for (int i = 0; i < cores; ++i)
    a = lp::max_f(a, x.GetValue(i * 16));
  y.SetValue(0, a);
}
extern "C" __global__ __aicore__ void
ascend_quant(GM_ADDR in, GM_ADDR data, GM_ADDR scales, GM_ADDR maximum, uint64_t rows,
             uint64_t cols, int dtype, int fmt, int block, bool tensor, bool stochastic,
             uint32_t seed) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> ibuf, obuf, sbuf;
  pipe.InitBuffer(ibuf, 4096);
  pipe.InitBuffer(obuf, 1024);
  pipe.InitBuffer(sbuf, 64);
  auto raw = ibuf.Get<uint8_t>(), bytes = obuf.Get<uint8_t>(),
       scalesLocal = sbuf.Get<uint8_t>();
  GlobalTensor<uint8_t> input, output, scaleout;
  input.SetGlobalBuffer(in);
  output.SetGlobalBuffer(data);
  scaleout.SetGlobalBuffer(scales);
  GlobalTensor<float> maxin;
  maxin.SetGlobalBuffer((__gm__ float *)maximum);
  float maximumValue = (fmt == lp::NVFP4 || tensor) ? maxin.GetValue(0) : 0;
  float global = fmt == lp::NVFP4 ? lp::global_scale(maximumValue) : 1;
  uint64_t gpr = (cols + block - 1) / block;
  unsigned groupsPerTile = 1024 / block;
  uint64_t tilesPerRow = (gpr + groupsPerTile - 1) / groupsPerTile;
  for (uint64_t tile = GetBlockIdx(); tile < rows * tilesPerRow;
       tile += GetBlockNum()) {
    uint64_t row = tile / tilesPerRow,
             firstGroup = (tile % tilesPerRow) * groupsPerTile,
             start = firstGroup * block;
    unsigned count = unsigned(
        cols - start < groupsPerTile * block ? cols - start : groupsPerTile * block);
    load_dma(raw, input[(row * cols + start) * (dtype == lp::FP32 ? 4 : 2)],
             count * (dtype == lp::FP32 ? 4 : 2));
    unsigned groups = (count + block - 1) / block;
    for (unsigned g = 0; g < groups; ++g) {
      unsigned first = g * block, end = first + block;
      if (end > count)
        end = count;
      float a = maximumValue;
      if (!tensor) {
        a = 0;
        for (unsigned j = first; j < end; ++j)
          a = lp::max_f(a, lp::abs_f(local_value(raw, j, dtype)));
      }
      uint8_t sc =
          fmt == lp::MXFP8 ? lp::mx_scale(a) : lp::encode8((a / global) / 6.0f);
      scalesLocal.SetValue(g, sc);
      float scale = lp::scale_value(sc, fmt);
      for (unsigned j = first; j < end; j += (fmt == lp::MXFP8 ? 1 : 2)) {
        uint8_t packed = 0;
        for (int k = 0; k < (fmt == lp::MXFP8 ? 1 : 2) && j + k < end; ++k) {
          uint64_t idx = row * cols + start + j + k;
          float v = local_value(raw, j + k, dtype);
          float u = stochastic ? lp::uniform(idx, seed) : 0;
          uint8_t code;
          if (fmt == lp::MXFP8)
            code = lp::encode8(lp::mx_scaled(v, sc), stochastic, u);
          else
            code = lp::encode4_scaled(v / global, scale, stochastic, u);
          packed |= code << (4 * k);
        }
        bytes.SetValue(fmt == lp::MXFP8 ? j : j / 2, packed);
      }
    }
    if (!tensor)
      store_dma(scaleout[row * gpr + firstGroup], scalesLocal, groups);
    else if (tile == 0)
      store_dma(scaleout, scalesLocal, 1);
    store_dma(output[fmt == lp::MXFP8 ? row * cols + start
                                      : row * ((cols + 1) / 2) + start / 2],
              bytes, fmt == lp::MXFP8 ? count : (count + 1) / 2);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
  }
}
// Direct cached GM reads outperform explicit UB prefetch for these small codes.
// Keep byte-granular DMA stores so neighboring rows cannot share scalar stores.
extern "C" __global__ __aicore__ void
ascend_dequant(GM_ADDR data, GM_ADDR scales, GM_ADDR out, float global, uint64_t rows,
               uint64_t cols, int fmt, int block, bool tensor, int outtype) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, 2048);
  auto bytes = buffer.Get<uint8_t>();
  auto f32 = buffer.Get<float>();
  auto f16 = buffer.Get<half>();
  auto b16 = buffer.Get<uint16_t>();
  GlobalTensor<uint8_t> codes, scalesin, output;
  codes.SetGlobalBuffer(data);
  scalesin.SetGlobalBuffer(scales);
  output.SetGlobalBuffer(out);
  uint64_t tiles = (cols + 511) / 512, gpr = (cols + block - 1) / block;
  for (uint64_t t = GetBlockIdx(); t < rows * tiles; t += GetBlockNum()) {
    uint64_t row = t / tiles, start = t % tiles * 512;
    uint32_t count = uint32_t(cols - start);
    if (count > 512)
      count = 512;
    for (uint32_t j = 0; j < count; ++j) {
      uint64_t c = start + j, g = tensor ? 0 : row * gpr + c / block;
      uint8_t code = codes.GetValue(fmt == lp::MXFP8 ? row * cols + c
                                                     : row * ((cols + 1) / 2) + c / 2);
      if (fmt == lp::NVFP4)
        code = (code >> ((c & 1) * 4)) & 15;
      float v = (fmt == lp::MXFP8 ? lp::fp8_value(code) : lp::fp4_value(code)) *
                lp::scale_value(scalesin.GetValue(g), fmt);
      v = v * global;
      if (outtype == lp::FP32)
        f32.SetValue(j, v);
      else if (outtype == lp::FP16)
        f16.SetValue(j, half(v));
      else
        b16.SetValue(j, lp::bf16(v));
    }
    store_dma(output[(row * cols + start) * (outtype == lp::FP32 ? 4 : 2)], bytes,
              count * (outtype == lp::FP32 ? 4 : 2));
  }
}
#ifndef __CCE_AICORE__
extern "C" void launch_ascend_amax(void *stream, void *in, void *scratch, void *maximum,
                                   uint64_t n, int dtype, unsigned cores) {
  ascend_amax<<<cores, nullptr, stream>>>((uint8_t *)in, (uint8_t *)scratch, n, dtype);
  ascend_max_finish<<<1, nullptr, stream>>>((uint8_t *)scratch, (uint8_t *)maximum,
                                            cores);
}
#endif
#ifndef __CCE_AICORE__
extern "C" void launch_ascend_quant(void *stream, void *in, void *data, void *scales,
                                    void *maximum, uint64_t rows, uint64_t cols,
                                    int dtype, int fmt, int block, bool tensor,
                                    bool stochastic, uint32_t seed, unsigned cores) {
  ascend_quant<<<cores, nullptr, stream>>>(
      (uint8_t *)in, (uint8_t *)data, (uint8_t *)scales, (uint8_t *)maximum, rows, cols,
      dtype, fmt, block, tensor, stochastic, seed);
}
#endif
#ifndef __CCE_AICORE__
extern "C" void launch_ascend_dequant(void *stream, void *data, void *scales, void *out,
                                      float global, uint64_t rows, uint64_t cols,
                                      int fmt, int block, bool tensor, int outtype,
                                      unsigned cores) {
  ascend_dequant<<<cores, nullptr, stream>>>((uint8_t *)data, (uint8_t *)scales,
                                             (uint8_t *)out, global, rows, cols, fmt,
                                             block, tensor, outtype);
}
#endif

// Standalone numeric checks share exactly the implementation used by kernels.
extern "C" __global__ __aicore__ void ascend_numeric(GM_ADDR ap, GM_ADDR bp,
                                                     GM_ADDR out, uint64_t base,
                                                     uint32_t seed, int mode) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> buf;
  pipe.InitBuffer(buf, 1024);
  auto f = buf.Get<float>();
  auto bytes = buf.Get<uint8_t>();
  GlobalTensor<float> a, b;
  a.SetGlobalBuffer((__gm__ float *)ap);
  b.SetGlobalBuffer((__gm__ float *)bp);
  GlobalTensor<uint8_t> y;
  y.SetGlobalBuffer(out);
  unsigned start = GetBlockIdx() * 256;
  for (unsigned j = 0; j < 256; ++j) {
    unsigned i = start + j;
    float v;
    if (mode == 0)
      v = lp::uniform(base + i, seed);
    else if (mode == 1)
      v = lp::multiply_rn(a.GetValue(i), b.GetValue(i));
    else
      v = lp::divide_rn(a.GetValue(i), b.GetValue(i));
    f.SetValue(j, v);
  }
  store_dma(y[start * 4], bytes, 1024);
}
#ifndef __CCE_AICORE__
extern "C" void launch_ascend_numeric(void *stream, void *a, void *b, void *out,
                                      uint64_t base, uint32_t seed, int mode) {
  ascend_numeric<<<16, nullptr, stream>>>((uint8_t *)a, (uint8_t *)b, (uint8_t *)out,
                                          base, seed, mode);
}
#endif
