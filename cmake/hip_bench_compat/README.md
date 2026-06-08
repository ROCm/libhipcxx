<!---
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
-->

# HIP benchmark compatibility shims

Fork-owned headers that let the **unmodified** upstream benchmark / nvbench_helper
sources (`benchmarks/**`, `nvbench_helper/**`) compile on HIP. Kept out of the
upstream-imported trees so re-cherry-picking those files never conflicts.

This directory is added to the include path of the benchmark targets on HIP by
`cccl_get_cub()` (see `cmake/CCCLGetDependencies.cmake`), so that:

- `#include <cub/...>`  -> forwards to `<hipcub/hipcub.hpp>` and aliases
  `namespace cub = hipcub` (only the few CUB headers the benches use are mirrored).
- `#include <curand.h>` -> forwards to `<hiprand/hiprand.h>` and maps the handful
  of `curand*` / `CURAND_*` symbols used by `nvbench_helper.cu` to their hipRAND
  equivalents.

rocThrust (`<thrust/...>`, incl. `thrust::cuda::par`) and hipCUB/hipRAND are
provided by ROCm directly, so no shim is needed for those.
