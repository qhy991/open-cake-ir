"""Read-only ranged fetch probe for the exact upstream Triton LLVM build."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from urllib.request import Request, urlopen

URL = "https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz"
CHUNK = 1 << 20
WORKERS = 8


def fetch(index: int) -> int:
    start = index * CHUNK
    end = start + CHUNK - 1
    request = Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers["Content-Range"].startswith(
            f"bytes {start}-{end}/"
        ):
            raise RuntimeError(f"Unexpected range response for chunk {index}")
        data = response.read()
    if len(data) != CHUNK:
        raise RuntimeError(f"Short ranged response for chunk {index}")
    return len(data)


if __name__ == "__main__":
    began = perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        total = sum(pool.map(fetch, range(WORKERS)))
    elapsed = perf_counter() - began
    print(f"{total} bytes / {elapsed:.2f} s = {total / elapsed / 1e6:.2f} MB/s")
