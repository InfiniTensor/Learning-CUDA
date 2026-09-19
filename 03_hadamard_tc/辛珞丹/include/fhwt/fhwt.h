// =============================================================================
//  fhwt.h -- public host interface of the fast Hadamard transform library.
//
//  The transform is applied to the *last* dimension of a contiguous
//  [rows, dim] tensor, which is exactly the layout of a
//  [batch, seq_len, num_heads, head_dim] activation tensor.
//
//      y[r, :] = scale * H_dim * x[r, :]
//
//  with H_dim the (unnormalised) Sylvester Hadamard matrix of order `dim`.
//  Passing scale = 1/sqrt(dim) yields the orthonormal transform used by
//  QuaRot / SpinQuant before quantisation.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "fhwt/backend.cuh"
#include <memory>

namespace fhwt {

enum class DType { kF16 = 0, kBF16 = 1, kF32 = 2 };

enum class KernelKind {
  kAuto = 0, // pick from dim / rows
  kReg = 1,  // register + warp shuffle
  kSmem = 2, // two-level: registers/shuffle + shared memory
  kTensorCore = 3,
  kQuant = 4, // fused transform + quantisation
};

// Quantisation formats supported by the fused path.
enum class QuantKind { kNone = 0, kFp8E4M3 = 1, kInt4 = 2 };

const char *to_string(DType t);
const char *to_string(KernelKind k);
const char *to_string(QuantKind q);

size_t dtype_size(DType t);

struct Config {
  DType dtype = DType::kF16;
  KernelKind kernel = KernelKind::kAuto;
  // 0 => library heuristic. threads_per_row must be a power of two <= 32.
  int threads_per_row = 0;
  int rows_per_block = 0;
  // Native packed accumulation is faster; fp32 accumulation matches the
  // reference implementation bit-for-bit in the accumulation order.
  bool fp32_accum = false;
  // 0 => exactly cover the work with one grid-stride iteration.
  int num_blocks = 0;
  // Fused quantisation.
  QuantKind quant = QuantKind::kNone;
  int block_size = 0; // scale block along dim (0 => per row); power of two in [8, dim]
};

// Concrete launch geometry chosen for a (rows, dim, config) triple.
struct LaunchPlan {
  int log_dim = 0;
  int threads_per_row = 0;
  int rows_per_block = 0;
  int block = 0;
  int grid = 0;
  int smem_bytes = 0;
  const char *kernel_name = "";
};

LaunchPlan plan(int64_t rows, int64_t dim, const Config &cfg);

// y = scale * H_dim(x);  x and y are contiguous [rows, dim] buffers.
void hadamard(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg, float scale,
              cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
//  Fused transform + per-row (or per-block) quantisation.
//  Quantised output layout:
//    kFp8E4M3 : quant[r, dim] (uint8) + scale[r] (float), one scale per row or
//               one per `block_size` columns (scales laid out row-major).
//    kInt4    : packed nibbles, two elements per byte along dim, plus one scale
//               per row.  Element j lives in the low nibble for even j.
//  The quantised tensor is stored with the same (rows, dim) shape.
// -----------------------------------------------------------------------------
void hadamard_quant(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                    const Config &cfg, float norm_scale, cudaStream_t stream = nullptr);

// Number of scale entries produced by hadamard_quant for the given config.
int64_t quant_scale_count(int64_t rows, int64_t dim, const Config &cfg);

// Standalone quantiser applied to an already-rotated tensor.  Together with
// hadamard() it forms the unfused reference pipeline
// ("rotate then quantise") used to validate the fused kernel.
void quantize_rows(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                   const Config &cfg, cudaStream_t stream = nullptr);

// -----------------------------------------------------------------------------
//  Reusable CUDA Graph for repeated transforms of the same shape.
//
//  Why: a small transform kernel runs for ~2.6 us while the CPU-side submission
//  of the next launch costs 7-30 us, so a loop of hadamard() calls is
//  submission-bound rather than kernel-bound (measured: the GPU is busy only
//  7-8% of the timing window, docs/profile/launch_bound_analysis.md).  Capturing
//  the whole burst once and replaying it with a single cudaGraphLaunch takes that
//  cost off the critical path.  Measured on the RTX 5060 Ti: 5-29x for small
//  shapes (dim 64/rows 1024 .. dim 128/rows 4096), 1.4-1.5x while the working set
//  sits in L2, and 1.00x once the kernel runs for hundreds of microseconds
//  (DRAM-bound).  See report section 8.5.
//
//  A graph captures buffer *addresses*, so x and y must stay alive and must not
//  be reallocated for as long as the graph is in use.  The main win comes from
//  putting many transforms into one graph; a single-node graph mainly avoids
//  repeating the setup on every call.
// -----------------------------------------------------------------------------
class Graph {
public:
  Graph() = default;
  ~Graph();
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  // Captures `repeats` identical transforms into one graph, replacing any
  // previously captured one.  `stream` is the stream the burst is recorded on;
  // when null the object creates (and owns) a private non-blocking stream.
  // Stream capture is illegal on the legacy default stream, so passing one is
  // rejected by CUDA itself.
  void capture(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg, float scale,
               int repeats, cudaStream_t stream = nullptr);

  // Same, for the fused transform + quantisation path.  Config::quant selects
  // the output format; Config::dtype must be fp16 / bf16 and dim must lie in
  // 32..1024 (the fused kernel's range).
  void capture_quant(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                     const Config &cfg, float norm_scale, int repeats,
                     cudaStream_t stream = nullptr);

  // Replays the captured burst.  `stream` defaults to the graph's own stream.
  void launch(cudaStream_t stream = nullptr) const;

  void reset();
  bool valid() const { return exec_ != nullptr; }
  int repeats() const { return repeats_; }
  // The stream the burst is recorded on and replayed on by default.  Recording
  // cudaEvents on this stream brackets a replay correctly.
  cudaStream_t stream() const { return stream_; }

private:
  void open_stream(cudaStream_t external);
  void close_capture();

  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool owns_stream_ = false;
  int repeats_ = 0;
};

// -----------------------------------------------------------------------------
//  GraphSequence -- many transforms, one launch.
//
//  Report section 8.5 has a corollary that is easy to get wrong: what makes a
//  CUDA Graph fast is the number of kernels *inside* it, not the fact that a
//  graph is used at all.  Replaying a graph that holds a single 2.6 us kernel
//  costs about as much as launching that kernel directly -- measured within
//  noise of 1.0x (0.93-3.59x over five sweeps, median 1.09x) when a serving
//  loop replayed one single-step graph per layer.  One graph holding the whole
//  layer sequence turns L submissions into 1.
//
//  A fork/join shape (add_parallel()) is a separate question and report
//  section 5.8 answers it: what costs is the number of *roots*, not the shape.
//  A forked graph whose steps are all roots replays ~4x slower than the
//  equivalent linear chain on this driver; the same nodes behind one serial
//  add() entry replay 1.07-1.12x *faster* than the chain.  add_parallel()
//  carries the recommendation that follows from that.
//
//  So the unit of capture is a *pass*: add() every transform the pass performs,
//  record() once, launch() per pass.  Steps may touch different buffers and mix
//  shapes, dtypes and the fused quantisation path; they are recorded in the
//  order they were added and replay on a single stream, so the result is
//  bit-identical to issuing every step directly.
//
//  The graph records buffer *addresses*: every x / y / scale_out handed to add()
//  must stay alive and must not be reallocated until reset() (or destruction).
// -----------------------------------------------------------------------------
class GraphSequence {
public:
  GraphSequence() = default;
  ~GraphSequence();
  GraphSequence(const GraphSequence &) = delete;
  GraphSequence &operator=(const GraphSequence &) = delete;

  // Queues one transform.  A config whose quant is not kNone is rejected --
  // use add_quant() for the fused path.
  void add(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg, float scale);

  // Queues one fused transform + quantisation step.
  void add_quant(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                 const Config &cfg, float norm_scale);

  // Queues a step that does NOT read anything a previous step in the same
  // sequence writes.  A sequence built only from add() replays as a linear
  // chain -- node i waits for node i-1 -- so a pass of L layers costs L kernel
  // durations however little of the GPU each layer's kernel actually fills.
  // add_parallel() declares the step independent of the preceding one and makes
  // record() emit a fork/join shape instead: the parallel siblings all wait on
  // the last non-parallel step, and the next non-parallel step waits for every
  // one of them.
  //
  // The caller owns that claim.  Two steps may be declared parallel only when
  // neither reads a buffer the other writes; a wrong claim is a data race, not
  // something the library can detect.  A correct claim cannot change the
  // answer: each step is a pure function of its own inputs and the non-parallel
  // steps keep their relative order.
  //
  // The caller owns a second thing here, and it is not obvious: a step queued
  // with add_parallel() that has no non-parallel step before it becomes a
  // *root* of the recorded graph.  On this driver (RTX 5060 Ti / Windows /
  // CUDA 12.9) a 32-layer pass recorded as 32 roots replays in 4.4-5.2 us per
  // step against 1.15 us for the linear chain, 0.24x, while the identical
  // nodes with layer 0 queued through add() -- one root, 31 forks -- replay in
  // 1.05-1.08 us, 1.07-1.12x *faster* than the chain and reproducibly so
  // (report 5.8 / 7.11).  The cost is host-side, it does not appear in the
  // kernel durations, and it grows over repeated replays of the same graph.
  // So for a pass of independent layers queue the first with add() and the
  // rest with add_parallel(): a fork/join fan-out is what to reach for, not a
  // flat set of roots.
  void add_parallel(const void *x, void *y, int64_t rows, int64_t dim, const Config &cfg,
                    float scale);
  void add_quant_parallel(const void *x, void *q_out, float *scale_out, int64_t rows, int64_t dim,
                          const Config &cfg, float norm_scale);

  // Records every queued step into one graph, replacing any previous recording.
  // `stream` is the recording stream and the default replay stream; when null
  // the object creates (and owns) a private non-blocking stream.
  void record(cudaStream_t stream = nullptr);

  // Replays the whole sequence.  Defaults to the recording stream.
  void launch(cudaStream_t stream = nullptr) const;

  void reset();
  int steps() const { return (int)steps_.size(); }
  // Number of steps queued with add_parallel() / add_quant_parallel().
  int forks() const { return forks_; }
  // How record() assembled a forked sequence: true when every step became a
  // kernel node in the parent graph (the captured one-node graph was unwrapped
  // with cudaGraphKernelNodeGetParams), false when the captured graphs had to be
  // added as child nodes.  Diagnostics only -- both forms replay identically --
  // but it is what distinguishes them in report 5.8 / 7.11.
  bool kernel_nodes() const { return kernel_nodes_; }
  bool recorded() const { return exec_ != nullptr; }
  cudaStream_t stream() const { return stream_; }
  // Raw handles so tests can inspect the recorded topology; both are null
  // before record().
  cudaGraph_t graph() const { return graph_; }
  cudaGraphExec_t exec() const { return exec_; }

private:
  struct Step {
    const void *x;
    void *y;
    float *scale_out;
    int64_t rows;
    int64_t dim;
    Config cfg;
    float scale;
    bool quant;
    bool independent;
  };

  void open_stream(cudaStream_t external);
  void drop_graph();
  void validate() const;
  void record_linear();
  void record_fork_join();
  void push(Step &&st);

  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  bool owns_stream_ = false;
  int forks_ = 0;
  bool kernel_nodes_ = false;
  std::vector<Step> steps_;
  // Single-step graphs used as child nodes by the fork/join recorder.  They are
  // referenced by the parent graph, so they stay alive until reset().
  std::vector<cudaGraph_t> children_;
};

// -----------------------------------------------------------------------------
//  GraphExecCache -- record each workload once, replay it many times.
//
//  A CUDA Graph bakes in the *addresses* of the buffers it was recorded with
//  (see Graph above) and building one costs ~0.13-0.23 ms for a 32-node pass on
//  this device (median 0.15 ms over seven configurations), so a serving loop
//  that keeps re-using the same activation
//  buffers wants the recorded graph looked up rather than re-recorded.
//
//  The key is the whole workload -- (x, y, scale_out, rows, dim, dtype, kernel,
//  threads_per_row, rows_per_block, num_blocks, fp32_accum, quant, block_size,
//  scale, repeats) -- not just the shape.  A request that matches a recorded
//  graph only in shape is a miss and gets recorded afresh.  That is deliberate:
//  replaying a graph whose pointers no longer match would silently transform
//  the wrong memory, and no amount of shape checking can catch it.  Callers
//  that reallocate between calls pay for a re-recording, which is the safe
//  behaviour; callers that keep their buffers get a hit.
//
//  An earlier revision of this header exposed a `GraphCache` that keyed on the
//  shape alone; measuring it showed the hit it reported was not a hit at all
//  once the buffers moved (a stale graph replayed, producing wrong output).
//  The address check below is the fix, and tests/test_graph.cu pins it.
//
//  Entries are evicted least-recently-used once capacity() is reached.
// -----------------------------------------------------------------------------
class GraphExecCache {
public:
  static constexpr int kDefaultCapacity = 64;
  explicit GraphExecCache(int capacity = kDefaultCapacity);
  ~GraphExecCache();
  GraphExecCache(const GraphExecCache &) = delete;
  GraphExecCache &operator=(const GraphExecCache &) = delete;

  // Pure query: would get_or_capture() find an executable graph without
  // recording?  Never touches the hit/miss counters.
  bool lookup(const void *x, void *y, const float *scale_out, int64_t rows, int64_t dim,
              const Config &cfg, float scale, int repeats) const;

  // Returns the executable graph for exactly this workload, recording it on
  // first use.  `scale_out` is null on the plain (non-fused) path.
  cudaGraphExec_t get_or_capture(const void *x, void *y, float *scale_out, int64_t rows,
                                 int64_t dim, const Config &cfg, float scale, int repeats);

  // get_or_capture() followed by cudaGraphLaunch on `stream` (null means the
  // cache's own stream).  This is what a serving loop calls.
  void launch(const void *x, void *y, float *scale_out, int64_t rows, int64_t dim,
              const Config &cfg, float scale, int repeats, cudaStream_t stream = nullptr);

  void clear();
  size_t size() const { return entries_.size(); }
  int capacity() const { return capacity_; }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t captures() const { return captures_; } // graphs actually recorded
  uint64_t evictions() const { return evictions_; }
  cudaStream_t stream() const { return stream_; }

private:
  struct Entry {
    const void *x;
    void *y;
    float *scale_out;
    int64_t rows;
    int64_t dim;
    Config cfg;
    float scale;
    int repeats;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    uint64_t stamp;
  };

  static bool same_config(const Config &a, const Config &b);
  static bool matches(const Entry &e, const void *x, void *y, const float *scale_out, int64_t rows,
                      int64_t dim, const Config &cfg, float scale, int repeats);
  void evict_one();

  std::vector<Entry> entries_;
  int capacity_;
  uint64_t stamp_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t captures_ = 0;
  uint64_t evictions_ = 0;
  cudaStream_t stream_ = nullptr;
};
// -----------------------------------------------------------------------------
//  Device property / bandwidth calibration helpers used by the benchmark tool.
// -----------------------------------------------------------------------------
struct DeviceInfo {
  char name[128];
  int major;
  int minor;
  int multi_processor_count;
  int max_threads_per_sm;
  int max_smem_per_sm;
  int clock_khz;
  int memory_clock_khz;
  int memory_bus_width;
  int l2_bytes;
};

DeviceInfo device_info(int device = 0);
double device_bandwidth_gbps(const DeviceInfo &info);

// -----------------------------------------------------------------------------
//  In-place advice.
//
//  Writing the transform back into the input buffer changes neither the DRAM
//  traffic nor the instruction count; it only halves the working set.  Measured
//  on this device that pays off *only* when the single-tensor footprint sits
//  between L2/2 and L2: there the out-of-place version needs twice the L2 and
//  falls back to DRAM (2.15x measured), while smaller tensors already fit either
//  way (1.00x) and larger ones are DRAM-bound either way (1.00x).
//  See report section 7.8.
// -----------------------------------------------------------------------------
bool inplace_pays_off(int64_t rows, int64_t dim, DType t, const DeviceInfo &info);

// -----------------------------------------------------------------------------
//  Tensor views and the aliasing half of the in-place decision.
//
//  inplace_pays_off(rows, dim, ...) above answers "is the transform cheap
//  enough" for a *contiguous* [rows, dim] buffer.  Real activation tensors are
//  not always contiguous: QuaRot rotates the last dim of a
//  [batch, heads, seq, head_dim] tensor, and both a tensor whose h/s strides
//  have been exchanged ("transposed layout", what the attention kernel wants)
//  and a head-sliced view show up in the same serving loop.
//
//  The kernels address a row as x + row * dim, so a view can be driven through
//  the flat API only when its last-dim slices *are* a contiguous [rows, dim]
//  buffer, visited in row-major order of the leading indices.  When they are
//  not, writing the result back into the input interleaves rows -- the row the
//  kernel writes is not the row it read -- and the transform is silently wrong.
//  The view overload therefore refuses instead of advising.
// -----------------------------------------------------------------------------
struct TensorView {
  const void *ptr = nullptr;
  // Logical [batch, heads, seq, dim]; unused leading entries must be 1.
  int64_t shape[4] = {1, 1, 1, 1};
  // Strides in elements.  A canonical contiguous [b, h, s, d] tensor has
  // {h*s*d, s*d, d, 1}.
  int64_t stride[4] = {0, 0, 0, 0};
};

// Number of rows a flattened view covers (batch * heads * seq) and its last-dim
// extent.
int64_t view_rows(const TensorView &v);
int64_t view_dim(const TensorView &v);

// True when the view's last-dim slices already form a contiguous [rows, dim]
// buffer, so hadamard() can be pointed straight at v.ptr and driven with
// view_rows() rows.  This is what keeps an in-place transform of a transposed,
// broadcast or strided view from scrambling the tensor.
bool row_contiguous(const TensorView &v);

// In-place advice for a view: the contiguity test above composed with the
// footprint test below.
bool inplace_pays_off(const TensorView &v, DType t, const DeviceInfo &info);

// Allocates nothing; returns the number of bytes touched by one transform.
inline double transfer_bytes(int64_t rows, int64_t dim, DType t) {
  return 2.0 * (double)rows * (double)dim * (double)dtype_size(t);
}

// -----------------------------------------------------------------------------
//  Bandwidth roof, measured with SIMT kernels rather than assumed.
//
//  `memcpy_*` are the driver's D2D copy path; `copy` / `read` / `write` are
//  grid-stride kernels doing 128-bit accesses, i.e. the same execution
//  resources the transform kernels use.  An out-of-place transform moves
//  2*rows*dim*sizeof(T) bytes, so `copy` is its roof.  On the MTT S4000 the
//  driver's theoretical figure (device_bandwidth_gbps) is *below* what the copy
//  path sustains, which is why the report normalises against these numbers and
//  reports the theoretical one only as context.
// -----------------------------------------------------------------------------
struct RoofResult {
  double memcpy_async_gbps = 0.0; // cudaMemcpyAsync DeviceToDevice
  double memcpy_sync_gbps = 0.0;  // cudaMemcpy      DeviceToDevice
  double copy_gbps = 0.0;         // vectorised SIMT copy (read + write)
  double read_gbps = 0.0;         // read-only reduction
  double write_gbps = 0.0;        // write-only store
  // The same copy on a buffer that fits in L2.  The ratio l2_copy/copy is the
  // factor "keeping the working set in cache" is worth on this part: ~4x on a
  // GB20x-class NVIDIA part, ~1x on the MTT S4000, which is why the in-place
  // transform pays off on one and does nothing on the other.
  double l2_copy_gbps = 0.0;
  double bytes = 0.0; // bytes per buffer
};

RoofResult bandwidth_roof(size_t bytes = 256ull << 20, int iters = 10, int warmup = 3);

// -----------------------------------------------------------------------------
//  Rotation fused into the consumer's load stage.
//
//  A QuaRot-style pipeline rotates an activation and then feeds it to a GEMM:
//
//      A_rot = A H_k        (M x K: read A, write A_rot)
//      C     = A_rot B^T    (M x N: read A_rot)
//
//  The rotation kernel is pure DRAM overhead: it reads A, writes a second copy
//  of A, and then the GEMM reads that copy back.  Fusing the rotation into the
//  GEMM's A-tile load removes both the write and the re-read -- the tile is
//  rotated in registers and shared memory on the way in.  Three passes over A
//  instead of four, 2*M*K*sizeof(T) bytes saved, and one kernel launch instead
//  of two.
//
//  What it does *not* do is make the GEMM faster.  The saved traffic is only
//  worth anything while the GEMM is not compute-bound, so --mode gemm measures
//  both regimes; the fused kernel wins in the skinny-N / large-M corner and
//  converges to 1.00x once the SIMT FLOPs dominate (report section 7.12).
// -----------------------------------------------------------------------------
struct GemmConfig {
  DType dtype = DType::kF16;
  // true : `a` holds the *unrotated* [M, K] activations, rotated on load.
  // false: `a` already holds a rotated tensor (the unfused pipeline's second
  //        pass), so the kernel only multiplies.  Both variants run the same
  //        GEMM core, which is what makes the comparison fair.
  bool fuse_rotation = true;
  // 0 asks for the library heuristic.  K must be a power of two in [128, 1024]
  // (the shared-memory tile holds a whole row of A); M and N may be arbitrary.
  int block_m = 0;
  int block_n = 0;
  // Splits the N range across gridDim.y.  1 makes every block own the whole row
  // of C (so each block re-reads all of B, which L2 absorbs); >1 shortens the
  // per-block N loop at the cost of loading the A tile once per split.
  int nsplit = 1;
  // Folded into the rotation; ignored when fuse_rotation is false.
  float scale = 1.0f;
};

// C[m, n] = (A H_k)[m, :] . B[n, :] with the Hadamard applied to the last dim
// of A.  A is [M, K] row-major, B is [N, K] row-major, C is [M, N] row-major,
// all in cfg.dtype.  fp32 accumulation, rounded to cfg.dtype on store.
void hadamard_gemm(const void *a, const void *b, void *c, int64_t m, int64_t n, int64_t k,
                   const GemmConfig &cfg, cudaStream_t stream = nullptr);

} // namespace fhwt
