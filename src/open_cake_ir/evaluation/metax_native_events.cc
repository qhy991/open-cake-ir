// The captured MACA runtime owns each API address. No callbacks or profiler run here.
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

extern "C" int cake_maca_event_cohort(
    void **api, void *function, const unsigned *dimensions, unsigned shared,
    void ***arguments, unsigned warmups, unsigned samples, void *reset,
    std::size_t reset_words, float *elapsed, unsigned *calls, unsigned *phase) {
  using Create = int (*)(void **, unsigned);
  using Record = int (*)(void *, void *);
  using Event = int (*)(void *);
  using Elapsed = int (*)(float *, void *, void *);
  using Reset = int (*)(std::uintptr_t, unsigned, std::size_t, void *);
  using Launch = int (*)(void *, unsigned, unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned, void *, void **, void *);
  const auto create = reinterpret_cast<Create>(api[0]);
  const auto record = reinterpret_cast<Record>(api[1]);
  const auto sync = reinterpret_cast<Event>(api[2]);
  const auto duration = reinterpret_cast<Elapsed>(api[3]);
  const auto destroy = reinterpret_cast<Event>(api[4]);
  const auto clear = reinterpret_cast<Reset>(api[5]);
  const auto launch = reinterpret_cast<Launch>(api[6]);
  const auto stream_sync = reinterpret_cast<Event>(api[7]);
  void *begin = nullptr, *end = nullptr;
  int status = 0;
  *calls = 0; *phase = 0;
#define CHECK(step, expr) do { *phase = step; status = (expr); if (status) goto finish; } while (0)
  CHECK(1, create(&begin, 0));
  CHECK(2, create(&end, 0));
  // Warm the reset and instantiate both event handles before any formal sample.
  CHECK(3, clear(reinterpret_cast<std::uintptr_t>(reset), 0x3f800000u, reset_words, nullptr));
  CHECK(4, record(begin, nullptr));
  CHECK(5, record(end, nullptr));
  CHECK(6, sync(end));
  for (unsigned i = 0; i < warmups + samples; ++i) {
    if (i >= warmups) {
      CHECK(7, clear(reinterpret_cast<std::uintptr_t>(reset), 0x3f800000u, reset_words, nullptr));
      CHECK(8, record(begin, nullptr));
    }
    CHECK(9, launch(function, dimensions[0], dimensions[1], dimensions[2],
                    dimensions[3], dimensions[4], dimensions[5], shared,
                    nullptr, arguments[i], nullptr));
    ++*calls;
    if (i + 1 == warmups) CHECK(10, stream_sync(nullptr));
    if (i >= warmups) {
      CHECK(11, record(end, nullptr));
      CHECK(12, sync(end));
      CHECK(13, duration(&elapsed[i - warmups], begin, end));
    }
  }
finish:
  // Synchronization and event cleanup occur even after a failed target launch.
  const int drained = stream_sync(nullptr);
  const int end_closed = end ? destroy(end) : 0;
  const int begin_closed = begin ? destroy(begin) : 0;
  if (!status && (drained || end_closed || begin_closed)) {
    *phase = 14;
    status = drained ? drained : end_closed ? end_closed : begin_closed;
  }
  return status;
#undef CHECK
}

namespace {
struct Gate {
  std::atomic<bool> release{false};
  std::atomic<bool> timeout{false};
};
void release_after_submission(void *opaque) {
  auto &gate = *static_cast<Gate *>(opaque);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!gate.release.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      gate.timeout.store(true, std::memory_order_release);
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}
}

extern "C" int cake_maca_gated_event_cohort(
    void **api, void *function, const unsigned *dimensions, unsigned shared,
    void ***arguments, unsigned warmups, unsigned samples, void *reset,
    std::size_t reset_words, float *elapsed, unsigned *calls, unsigned *phase) {
  using Create = int (*)(void **, unsigned);
  using Record = int (*)(void *, void *);
  using Event = int (*)(void *);
  using Elapsed = int (*)(float *, void *, void *);
  using Reset = int (*)(std::uintptr_t, unsigned, std::size_t, void *);
  using Launch = int (*)(void *, unsigned, unsigned, unsigned, unsigned,
                         unsigned, unsigned, unsigned, void *, void **, void *);
  using Wait = int (*)(void *, void *, unsigned);
  using Host = int (*)(void *, void (*)(void *), void *);
  std::unique_ptr<Gate[]> gates(new Gate[warmups + samples]);
  void *begin=nullptr, *end=nullptr, *ready=nullptr, *stream=nullptr, *gate_stream=nullptr;
  int status=0; *calls=0; *phase=0;
#define GATED_CHECK(step, expr) do { *phase=step; status=(expr); if (status) goto finish; } while (0)
  GATED_CHECK(1,reinterpret_cast<Create>(api[0])(&begin,0));
  GATED_CHECK(2,reinterpret_cast<Create>(api[0])(&end,0));
  // Both constants are declared by the captured MACA SDK.
  GATED_CHECK(3,reinterpret_cast<Create>(api[0])(&ready,2)); // mcEventDisableTiming
  GATED_CHECK(4,reinterpret_cast<Create>(api[8])(&stream,1)); // mcStreamNonBlocking
  GATED_CHECK(5,reinterpret_cast<Create>(api[8])(&gate_stream,1));
  for (unsigned i=0;i<warmups+samples;++i) {
    GATED_CHECK(6,reinterpret_cast<Reset>(api[5])(
        reinterpret_cast<std::uintptr_t>(reset),0x3f800000u,reset_words,stream));
    // The ready event is queued behind a host callback that touches no device API.
    // The target stream waits until begin/target/end have all been submitted.
    GATED_CHECK(7,reinterpret_cast<Host>(api[11])(gate_stream,release_after_submission,&gates[i]));
    GATED_CHECK(8,reinterpret_cast<Record>(api[1])(ready,gate_stream));
    GATED_CHECK(9,reinterpret_cast<Wait>(api[10])(stream,ready,0));
    GATED_CHECK(10,reinterpret_cast<Record>(api[1])(begin,stream));
    GATED_CHECK(11,reinterpret_cast<Launch>(api[6])(function,
        dimensions[0],dimensions[1],dimensions[2],dimensions[3],dimensions[4],dimensions[5],
        shared,stream,arguments[i],nullptr));
    ++*calls;
    GATED_CHECK(12,reinterpret_cast<Record>(api[1])(end,stream));
    gates[i].release.store(true,std::memory_order_release);
    GATED_CHECK(13,reinterpret_cast<Event>(api[2])(end));
    if (gates[i].timeout.load(std::memory_order_acquire)) {
      status=-30001; *phase=14; goto finish;
    }
    if (i>=warmups)
      GATED_CHECK(15,reinterpret_cast<Elapsed>(api[3])(&elapsed[i-warmups],begin,end));
  }
finish:
  for (unsigned i=0;i<warmups+samples;++i) gates[i].release.store(true,std::memory_order_release);
  int cleanup=gate_stream ? reinterpret_cast<Event>(api[7])(gate_stream) : 0;
  if (stream) { int code=reinterpret_cast<Event>(api[7])(stream); if (!cleanup) cleanup=code; }
  for (void *event : {begin,end,ready}) if (event) {
    int code=reinterpret_cast<Event>(api[4])(event); if (!cleanup) cleanup=code;
  }
  for (void *owned : {stream,gate_stream}) if (owned) {
    int code=reinterpret_cast<Event>(api[9])(owned); if (!cleanup) cleanup=code;
  }
  if (!status && cleanup) { status=cleanup; *phase=16; }
  return status;
#undef GATED_CHECK
}
