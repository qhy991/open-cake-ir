// The captured MACA runtime owns each API address. No callbacks or profiler run here.
#include <cstddef>
#include <cstdint>

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
