# Explicit reference corrections

Original supplied source files are retained unchanged. A patch here is a separately
identified derived reference, never a relabelled original champion or inherited score.

`003_graph_parameter_order.patch` repairs only the host Graph parameter list. The
actual CUDA kernel takes four pointers and `float eps`; the original update list
inserts `int M` before eps. After a node update, M bits are interpreted as epsilon.
The original C++20 build failed all 458752 output elements on the zeros validation
case, with inputs preserved. The patch removes that extra parameter and keeps the
CUDA device function text unchanged. It must pass a fresh all-case GPU check before
its performance is compared. The original failure remains retained.

C++20 is an explicit build-compatibility treatment under NVCC 13.3 / Torch 2.11,
not a source rewrite or a claim that the original C++17 build succeeded. Its two-arm
CPU object-build probe and all subsequent input metadata retain that distinction.
