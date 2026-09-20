// MUPTI activity collector. Load only for profiling; never for benchmarks.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mupti.h>
#include <musa_runtime_api.h>
#include <pthread.h>

namespace {
FILE *output = nullptr;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
bool enabled = false;
void check(MUptiResult status, const char *call) {
  if (status != MUPTI_SUCCESS) {
    const char *message = nullptr;
    muptiGetResultString(status, &message);
    std::fprintf(stderr, "MUPTI %s: %s (%d)\n", call, message ? message : "unknown",
                 int(status));
    std::exit(2);
  }
}
void MUPTIAPI request(uint8_t **buffer, size_t *size, size_t *records) {
  *size = 4 * 1024 * 1024;
  *records = 0;
  *buffer = static_cast<uint8_t *>(std::malloc(*size));
  if (!*buffer)
    std::abort();
}
void MUPTIAPI complete(MUcontext context, uint32_t stream, uint8_t *buffer, size_t,
                       size_t valid) {
  MUpti_Activity *record = nullptr;
  pthread_mutex_lock(&lock);
  while (true) {
    MUptiResult status = muptiActivityGetNextRecord(buffer, valid, &record);
    if (status == MUPTI_ERROR_MAX_LIMIT_REACHED)
      break;
    check(status, "get record");
    if (record->kind == MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        record->kind == MUPTI_ACTIVITY_KIND_KERNEL) {
      const auto *k = reinterpret_cast<const MUpti_ActivityKernel6 *>(record);
      std::fprintf(
          output,
          "%" PRIu64 "\t%" PRIu64 "\t%u\t%u\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%d\t%d\t%s\n",
          k->start, k->end, k->deviceId, k->streamId, k->gridX, k->gridY, k->gridZ,
          k->blockX, k->blockY, k->blockZ, unsigned(k->registersPerThread),
          k->staticSharedMemory, k->dynamicSharedMemory, k->name ? k->name : "unknown");
    }
  }
  size_t dropped = 0;
  check(muptiActivityGetNumDroppedRecords(context, stream, &dropped),
        "dropped records");
  std::fprintf(output, "# dropped_records=%zu\n", dropped);
  std::fflush(output);
  pthread_mutex_unlock(&lock);
  std::free(buffer);
}
__attribute__((constructor)) void begin() {
  const char *path = std::getenv("LP_MUPTI_OUTPUT");
  if (!path)
    return;
  output = std::fopen(path, "w");
  if (!output) {
    std::perror(path);
    std::exit(2);
  }
  std::fprintf(
      output,
      "start_ns\tend_ns\tdevice\tstream\tgrid_x\tgrid_y\tgrid_z\tblock_x\tblock_"
      "y\tblock_z\tregisters\tstatic_shared\tdynamic_shared\tkernel\n");
  check(muptiActivityRegisterCallbacks(request, complete), "register callbacks");
  check(muptiActivityEnable(MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),
        "enable kernel activities");
  enabled = true;
}
__attribute__((destructor)) void end() {
  if (!enabled)
    return;
  musaDeviceSynchronize();
  check(muptiActivityFlushAll(1), "flush");
  std::fclose(output);
}
} // namespace
