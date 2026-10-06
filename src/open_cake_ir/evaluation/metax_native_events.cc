// The captured MACA runtime owns each API address. No callbacks or profiler run here.
#include <cstddef>
#include <cstdint>
#include <vector>

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

// Each captured graph contains one reset, two events and exactly one target launch.
// Capture and instantiation precede all timed execution. Arguments remain distinct.
extern "C" int cake_maca_graph_event_cohort(
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
  using Capture = int (*)(void *, int);
  using EndCapture = int (*)(void *, void **);
  using Instantiate = int (*)(void **, void *, void **, char *, std::size_t);
  using GraphLaunch = int (*)(void *, void *);
  using RecordFlags = int (*)(void *, void *, unsigned);
  void *begin = nullptr, *end = nullptr, *stream = nullptr;
  std::vector<void *> graphs(warmups + samples, nullptr), executions(warmups + samples, nullptr);
  int status = 0; bool capturing = false;
  *calls = 0; *phase = 0;
#define GRAPH_CHECK(step, expr) do { *phase = step; status = (expr); if (status) goto finish; } while (0)
  GRAPH_CHECK(1, reinterpret_cast<Create>(api[0])(&begin, 0));
  GRAPH_CHECK(2, reinterpret_cast<Create>(api[0])(&end, 0));
  // mcStreamNonBlocking=0x01 is declared by the captured MACA SDK.
  GRAPH_CHECK(15, reinterpret_cast<Create>(api[8])(&stream, 1));
  GRAPH_CHECK(4, reinterpret_cast<Record>(api[1])(begin, stream));
  GRAPH_CHECK(5, reinterpret_cast<Record>(api[1])(end, stream));
  GRAPH_CHECK(6, reinterpret_cast<Event>(api[2])(end));
  for (unsigned i = 0; i < warmups + samples; ++i) {
    GRAPH_CHECK(16, reinterpret_cast<Capture>(api[10])(stream, 0));
    capturing = true;
    GRAPH_CHECK(17, reinterpret_cast<Reset>(api[5])(
        reinterpret_cast<std::uintptr_t>(reset), 0x3f800000u, reset_words, stream));
    // mcEventRecordExternal=0x01 creates real event-record graph nodes.
    GRAPH_CHECK(18, reinterpret_cast<RecordFlags>(api[16])(begin, stream, 1));
    GRAPH_CHECK(19, reinterpret_cast<Launch>(api[6])(function,
        dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4], dimensions[5],
        shared, stream, arguments[i], nullptr));
    GRAPH_CHECK(20, reinterpret_cast<RecordFlags>(api[16])(end, stream, 1));
    GRAPH_CHECK(21, reinterpret_cast<EndCapture>(api[11])(stream, &graphs[i]));
    capturing = false;
    GRAPH_CHECK(22, reinterpret_cast<Instantiate>(api[12])(&executions[i], graphs[i], nullptr, nullptr, 0));
  }
  for (unsigned i = 0; i < warmups + samples; ++i) {
    GRAPH_CHECK(23, reinterpret_cast<GraphLaunch>(api[13])(executions[i], stream));
    ++*calls;
    GRAPH_CHECK(24, reinterpret_cast<Event>(api[2])(end));
    if (i >= warmups)
      GRAPH_CHECK(25, reinterpret_cast<Elapsed>(api[3])(&elapsed[i - warmups], begin, end));
  }
finish:
  if (capturing) {
    void *unfinished = nullptr;
    reinterpret_cast<EndCapture>(api[11])(stream, &unfinished);
    if (unfinished) reinterpret_cast<Event>(api[14])(unfinished);
  }
  int cleanup = stream ? reinterpret_cast<Event>(api[7])(stream) : 0;
  for (void *exec : executions) if (exec) {
    const int code = reinterpret_cast<Event>(api[15])(exec); if (!cleanup) cleanup = code;
  }
  for (void *graph : graphs) if (graph) {
    const int code = reinterpret_cast<Event>(api[14])(graph); if (!cleanup) cleanup = code;
  }
  if (end) { const int code = reinterpret_cast<Event>(api[4])(end); if (!cleanup) cleanup = code; }
  if (begin) { const int code = reinterpret_cast<Event>(api[4])(begin); if (!cleanup) cleanup = code; }
  if (stream) { const int code = reinterpret_cast<Event>(api[9])(stream); if (!cleanup) cleanup = code; }
  if (!status && cleanup) { *phase = 26; status = cleanup; }
  return status;
#undef GRAPH_CHECK
}
