#pragma once

// The caller admits 1 <= chunks <= Tokens and 0 <= token < Tokens.
#if defined(__CUDACC__)
#define CAKE_WEAVE_HD __host__ __device__ __forceinline__
#else
#define CAKE_WEAVE_HD inline
#endif

namespace cake_weave {

template<int Tokens>
CAKE_WEAVE_HD int chunk_for_token(int token, int chunks) {
    const int base = Tokens / chunks;
    const int longer = Tokens % chunks;
    const int long_span = longer * (base + 1);
    return token < long_span ? token / (base + 1)
                             : longer + (token - long_span) / base;
}

template<int Tokens>
CAKE_WEAVE_HD int chunk_size(int chunk, int chunks) {
    return Tokens / chunks + (chunk < Tokens % chunks);
}

}  // namespace cake_weave

#undef CAKE_WEAVE_HD
