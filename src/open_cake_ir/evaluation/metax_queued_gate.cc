// Unadmitted component: queue a complete callback between stream-0 events.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <thread>

namespace {
struct Gate {
  std::atomic<bool> release{false};
  std::atomic<bool> timed_out{false};
};
void wait_for_submission(void *pointer) {
  auto &gate = *static_cast<Gate *>(pointer);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!gate.release.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      gate.timed_out.store(true, std::memory_order_release);
      return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}
}

// api: event create/record/sync/elapsed/destroy, stream create/destroy/wait,
// host callback, stream sync. reset and launch preserve their existing owners.
// completed_calls counts whole successful callbacks, not Program stage launches.
extern "C" int cake_default_stream_cohort(
    void **api, int (*reset)(void *), int (*launch)(void *, unsigned), void *user,
    unsigned warmups, unsigned samples, float *elapsed, unsigned *completed_calls,
    unsigned *phase, int *cleanup_status, unsigned *drained) {
  if (!api || !reset || !launch || !elapsed || !completed_calls || !phase ||
      !cleanup_status || !drained || warmups != 11 || samples != 5) return -41000;
  for (unsigned i = 0; i < 10; ++i) if (!api[i]) return -41000;
  using Create = int (*)(void **, unsigned);
  using Record = int (*)(void *, void *);
  using One = int (*)(void *);
  using Elapsed = int (*)(float *, void *, void *);
  using Wait = int (*)(void *, void *, unsigned);
  using Host = int (*)(void *, void (*)(void *), void *);
  auto gates = std::make_unique<Gate[]>(samples);
  void *begin = nullptr, *end = nullptr, *ready = nullptr, *gate_stream = nullptr;
  bool default_touched = false;
  int status = 0;
  *completed_calls = *phase = *drained = 0;
  *cleanup_status = 0;
#define CHECK(step, expression) do { *phase = step; status = (expression); if (status) goto finish; } while (0)
  CHECK(1, reinterpret_cast<Create>(api[0])(&begin, 0));
  CHECK(2, reinterpret_cast<Create>(api[0])(&end, 0));
  CHECK(3, reinterpret_cast<Create>(api[0])(&ready, 2)); // mcEventDisableTiming
  CHECK(4, reinterpret_cast<Create>(api[5])(&gate_stream, 1)); // nonblocking
  default_touched = true;
  CHECK(5, reinterpret_cast<Record>(api[1])(begin, nullptr));
  CHECK(6, reinterpret_cast<Record>(api[1])(end, nullptr));
  CHECK(7, reinterpret_cast<One>(api[2])(end));
  CHECK(8, reset(user)); // Match one reset before all eleven ordinary warmups.
  for (unsigned i = 0; i < warmups; ++i) {
    CHECK(9, launch(user, i));
    ++*completed_calls;
  }
  CHECK(10, reinterpret_cast<One>(api[9])(nullptr));
  for (unsigned i = 0; i < samples; ++i) {
    CHECK(11, reset(user));
    CHECK(12, reinterpret_cast<Host>(api[8])(gate_stream, wait_for_submission, &gates[i]));
    CHECK(13, reinterpret_cast<Record>(api[1])(ready, gate_stream));
    CHECK(14, reinterpret_cast<Wait>(api[7])(nullptr, ready, 0));
    CHECK(15, reinterpret_cast<Record>(api[1])(begin, nullptr));
    CHECK(16, launch(user, warmups + i)); // All Program stages remain in this call.
    ++*completed_calls;
    CHECK(17, reinterpret_cast<Record>(api[1])(end, nullptr));
    gates[i].release.store(true, std::memory_order_release);
    CHECK(18, reinterpret_cast<One>(api[2])(end));
    if (gates[i].timed_out.load(std::memory_order_acquire)) {
      status = -41001; *phase = 19; goto finish;
    }
    CHECK(20, reinterpret_cast<Elapsed>(api[3])(&elapsed[i], begin, end));
  }
finish:
  for (unsigned i = 0; i < samples; ++i) gates[i].release.store(true, std::memory_order_release);
  {
    const int gate_drain = gate_stream ? reinterpret_cast<One>(api[9])(gate_stream) : 0;
    const int target_drain = default_touched ? reinterpret_cast<One>(api[9])(nullptr) : 0;
    *cleanup_status = gate_drain ? gate_drain : target_drain;
    *drained = *cleanup_status == 0;
    if (!*drained) {
      // A failed drain gives no permission to free memory a callback can still use.
      // Retain gates and runtime resources until this diagnostic process exits.
      gates.release();
    } else {
      for (void *event : {begin, end, ready}) if (event) {
        const int code = reinterpret_cast<One>(api[4])(event);
        if (!*cleanup_status) *cleanup_status = code;
      }
      if (gate_stream) {
        const int code = reinterpret_cast<One>(api[6])(gate_stream);
        if (!*cleanup_status) *cleanup_status = code;
      }
    }
    if (!status && *cleanup_status) { status = *cleanup_status; *phase = 21; }
  }
  return status;
#undef CHECK
}
