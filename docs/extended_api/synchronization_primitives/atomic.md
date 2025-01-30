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
# `cuda::atomic`

Defined in header `<cuda/atomic>`:

```hip
template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
class cuda::atomic;
```

The class template `cuda::atomic` is an extended form of [`cuda::std::atomic`]
  that takes an additional [`cuda::thread_scope`] argument, defaulted to
  `cuda::std::thread_scope_system`.
It has the same interface and semantics as [`cuda::std::atomic`], with the
  following additional operations.

## Atomic Fence Operations

| Name                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| [`cuda::atomic_thread_fence`] | Memory order and scope dependent fence synchronization primitive. `(function)` |

## Atomic Extrema Operations

| Name                       | Description                                                                               |
|----------------------------|-------------------------------------------------------------------------------------------|
| [`cuda::atomic::fetch_min`] | Atomically find the minimum of the stored value and a provided value. `(member function)` |
| [`cuda::atomic::fetch_max`] | Atomically find the maximum of the stored value and a provided value. `(member function)` |

## Concurrency Restrictions

An object of type `cuda::atomic` or [`cuda::std::atomic`] shall not be accessed
  concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess`] property is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported`] property is 1.

Note, for objects of scopes other than `cuda::thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

## Implementation-Defined Behavior

For each type `T` and [`cuda::thread_scope`] `S`, the value of
  `cuda::atomic<T, S>::is_always_lock_free()` is as follows:

| Type `T` | [`cuda::thread_scope`] `S` | `cuda::atomic<T, S>::is_always_lock_free()` |
|----------|----------------------------|---------------------------------------------|
| Any      | Any                        | `sizeof(T) <= 8`                            |

## Example

```hip
#include <cuda/atomic>

__global__ void example_kernel() {
  // This atomic is suitable for all threads in the system.
  cuda::atomic<int, cuda::thread_scope_system> a;

  // This atomic has the same type as the previous one (`a`).
  cuda::atomic<int> b;

  // This atomic is suitable for all threads on the current processor (e.g. GPU).
  cuda::atomic<int, cuda::thread_scope_device> c;

  // This atomic is suitable for threads in the same thread block.
  cuda::atomic<int, cuda::thread_scope_block> d;
}
```


[`cuda::thread_scope`]: ../memory_model.md

[`cuda::atomic_thread_fence`]: ./atomic/atomic_thread_fence.md

[`cuda::atomic::fetch_min`]: ./atomic/fetch_min.md
[`cuda::atomic::fetch_max`]: ./atomic/fetch_max.md

[`cuda::std::atomic`]: https://en.cppreference.com/w/cpp/atomic/atomic

[`cuda::atomic`]: ./atomic.md

[atomics.types.int]: https://eel.is/c++draft/atomics.types.int
[atomics.types.pointer]: https://eel.is/c++draft/atomics.types.pointer

[`concurrentManagedAccess`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#abfea758a5672cb7803ac467543c11b67
[`hostNativeAtomicSupported`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#a64696b1aa1789ca322e8c86b69b57e7c
