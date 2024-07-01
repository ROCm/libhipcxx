<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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
# `cuda::atomic::fetch_max`

Defined in header `<cuda/atomic>`:

```cuda
template <typename T, cuda::thread_scope Scope>
__host__ __device__
T cuda::atomic<T, Scope>::fetch_max(T const& val,
                                    cuda::std::memory_order order
                                      = cuda::std::memory_order_seq_cst);
```

Atomically find the maximum of the value stored in the `cuda::atomic` and `val`.
The maximum is found using [`cuda::std::max`].

## Example

```cuda
#include <cuda/atomic>

__global__ void example_kernel() {
  cuda::atomic<int> a(0);
  auto x = a.fetch_max(1);
  auto y = a.load();
  assert(x == 0 && y == 1);
}
```


[`cuda::std::max`]: https://en.cppreference.com/w/cpp/algorithm/max
