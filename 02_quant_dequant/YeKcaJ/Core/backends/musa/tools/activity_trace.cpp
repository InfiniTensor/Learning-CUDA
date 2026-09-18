// 可选 MUPTI 采集插件；通过 LD_PRELOAD 使用，不参与正式量化程序。
#include <mupti.h>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

namespace {
FILE* output = nullptr;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
unsigned long long records = 0, dropped = 0;

void check(MUptiResult status, const char* operation) {
  if (status == MUPTI_SUCCESS) return;
  const char* message = nullptr;
  muptiGetResultString(status, &message);
  std::fprintf(stderr, "MUPTI %s failed: %s (%d)\n", operation,
               message ? message : "unknown", int(status));
  std::exit(2);
}

void MUPTIAPI request(uint8_t** buffer, size_t* size, size_t* max_records) {
  *size = 1 << 20;
  *max_records = 0;
  void* allocation = nullptr;
  if (posix_memalign(&allocation, 8, *size)) std::abort();
  *buffer = static_cast<uint8_t*>(allocation);
}

void MUPTIAPI complete(MUcontext context, uint32_t stream, uint8_t* buffer,
                       size_t, size_t valid_size) {
  pthread_mutex_lock(&lock);
  MUpti_Activity* record = nullptr;
  MUptiResult status;
  while ((status = muptiActivityGetNextRecord(buffer, valid_size, &record)) == MUPTI_SUCCESS) {
    if (record->kind != MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL &&
        record->kind != MUPTI_ACTIVITY_KIND_KERNEL) continue;
    const auto* k = reinterpret_cast<const MUpti_ActivityKernel6*>(record);
    std::fprintf(output, "%llu\t%llu\t%d\t%d\t%u\t%d\t%u\t%s\n",
                 static_cast<unsigned long long>(k->start),
                 static_cast<unsigned long long>(k->end), k->gridX, k->blockX,
                 unsigned(k->registersPerThread), k->staticSharedMemory,
                 k->localMemoryPerThread, k->name ? k->name : "unknown");
    ++records;
  }
  if (status != MUPTI_ERROR_MAX_LIMIT_REACHED) check(status, "read record");
  size_t lost = 0;
  check(muptiActivityGetNumDroppedRecords(context, stream, &lost), "dropped records");
  dropped += lost;
  pthread_mutex_unlock(&lock);
  std::free(buffer);
}

__attribute__((constructor)) void start() {
  const char* path = std::getenv("LP_MUPTI_TRACE");
  if (!path) return;
  output = std::fopen(path, "wx");
  if (!output) { std::perror("LP_MUPTI_TRACE"); std::exit(2); }
  std::fputs("start_ns\tend_ns\tgrid_x\tblock_x\tregisters\tshared_bytes\tlocal_bytes\tname\n", output);
  check(muptiActivityRegisterCallbacks(request, complete), "register callbacks");
  check(muptiActivityEnable(MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL), "enable kernels");
}

__attribute__((destructor)) void finish() {
  if (!output) return;
  check(muptiActivityFlushAll(0), "flush");
  std::fprintf(stderr, "MUPTI kernel_records=%llu dropped=%llu\n", records, dropped);
  std::fclose(output);
}
}  // namespace
