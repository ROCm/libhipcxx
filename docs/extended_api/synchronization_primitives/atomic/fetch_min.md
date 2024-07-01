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

# `hip::atomic::fetch_min`

Defined in header `<hip/atomic>`:

```hip
template <typename T, hip::thread_scope Scope>
__host__ __device__
T hip::atomic<T, Scope>::fetch_min(T const& val,
                                    hip::std::memory_order order
                                      = hip::std::memory_order_seq_cst);
```

Atomically find the minimum of the value stored in the `hip::atomic` and `val`.
The minimum is found using [`hip::std::min`].

## Example

```hip
#include <hip/atomic>

__global__ void example_kernel() {
  hip::atomic<int> a(1);
  auto x = a.fetch_min(0);
  auto y = a.load();
  assert(x == 1 && y == 0);
}
```


[`hip::std::min`]: https://en.cppreference.com/w/cpp/algorithm/min
