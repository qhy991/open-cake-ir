// Native timestamp controls for MCPTI diagnosis. No GPU calls or runtime imports.
// Diagnostic callbacks only: this file does not change the production SDK timer.
#include <atomic>
#include <cstdint>
#include <ctime>

static uint64_t clock_ns(clockid_t clock) {
    timespec stamp;
    if (clock_gettime(clock, &stamp) != 0) return 0;
    return uint64_t(stamp.tv_sec) * 1000000000ULL + uint64_t(stamp.tv_nsec);
}
extern "C" uint64_t cake_wall_time() { return clock_ns(CLOCK_REALTIME); }
extern "C" uint64_t cake_monotonic() { return clock_ns(CLOCK_MONOTONIC); }
extern "C" uint64_t cake_relative_monotonic() {
    static std::atomic<uint64_t> origin{0};
    uint64_t stamp = clock_ns(CLOCK_MONOTONIC);
    if (!stamp) return 0;
    uint64_t empty = 0;
    origin.compare_exchange_strong(empty, stamp);
    return stamp - origin.load() + 1;
}
