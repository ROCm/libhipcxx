<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
-->

# `cuda::atomic::atomic_thread_fence`

Defined in header `<cuda/atomic>`:

```hip
__host__ __device__
void cuda::atomic_thread_fence(cuda::std::memory_order order,
                               cuda::thread_scope scope = cuda::thread_scope_system);
```

Establishes memory synchronization ordering of non-atomic and relaxed atomic
  accesses, as instructed by `order`, for all threads within `scope` without an
  associated atomic operation.
It has the same semantics as [`cuda::std::atomic_thread_fence`].

## Example

```hip
#include <cuda/atomic>

__global__ void example_kernel(int* data) {
  *data = 42;
  cuda::atomic_thread_fence(cuda::std::memory_order_release,
                            cuda::thread_scope_device);
}
```

[`cuda::std::atomic_thread_fence`]: https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence
