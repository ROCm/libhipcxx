..
    MIT License

    Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

.. meta::
  :description: API reference for cuda::discard_memory, which discards modified cache lines of a global memory range without writing them back in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, discard_memory, cache, global memory, scratch memory

.. _libcudacxx-extended-api-memory-discard-memory:

``cuda::discard_memory``
========================

This page documents ``cuda::discard_memory``, which hints that a range of global memory no longer needs to be written back from the cache.

Defined in header ``<cuda/memory>``, ``<cuda/discard_memory>`` (deprecated since libhipcxx 3.4).

.. code:: cpp

    __host__ __device__
    void discard_memory(volatile void* ptr, size_t nbytes);

Discard modified cache lines without writing back the cached data to memory. The functionality enables using global memory as temporary scratch space. Does **not** generate any HW store operations.

Equivalent to ``memset(ptr, _indeterminate_, nbytes)``.

**Preconditions**

- ``ptr`` points to a valid allocation in *global memory* of size greater or equal to ``nbytes``.

Example
-------

This kernel needs a scratch pad that does not fit in shared memory, so it uses an allocation in global memory instead:

.. code:: cpp

   #include <cuda/memory>

    __device__ int compute(int* scratch, size_t N);

    __global__ void kernel(const int* in, int* out, int* scratch, size_t N) {
        // Each thread reads N elements into the scratch pad:
        for (int i = 0; i < N; ++i) {
            int idx      = threadIdx.x + i * blockDim.x;
            scratch[idx] = in[idx];
        }
        __syncthreads();
        // All threads compute on the scratch pad:
        int result = compute(scratch, N);
        // All threads discard the scratch pad memory to _hint_ that it does not need to be flushed from the cache:
        cuda::discard_memory(scratch + threadIdx.x * N, N * sizeof(int));
        __syncthreads();
        out[threadIdx.x] = result;
    }
