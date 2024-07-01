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
# `hip::atomic`

Defined in header `<hip/atomic>`:

```hip
template <typename T, hip::thread_scope Scope = hip::thread_scope_system>
class hip::atomic;
```

The class template `hip::atomic` is an extended form of [`hip::std::atomic`]
  that takes an additional [`hip::thread_scope`] argument, defaulted to
  `hip::std::thread_scope_system`.
It has the same interface and semantics as [`hip::std::atomic`], with the
  following additional operations.

## Atomic Fence Operations

| Name                         | Description                                                                    |
|------------------------------|--------------------------------------------------------------------------------|
| [`hip::atomic_thread_fence`] | Memory order and scope dependent fence synchronization primitive. `(function)` |

## Atomic Extrema Operations

| Name                       | Description                                                                               |
|----------------------------|-------------------------------------------------------------------------------------------|
| [`hip::atomic::fetch_min`] | Atomically find the minimum of the stored value and a provided value. `(member function)` |
| [`hip::atomic::fetch_max`] | Atomically find the maximum of the stored value and a provided value. `(member function)` |

## Concurrency Restrictions

An object of type `hip::atomic` or [`hip::std::atomic`] shall not be accessed
  concurrently by CPU and GPU threads unless:
- it is in unified memory and the [`concurrentManagedAccess`] property is 1, or
- it is in CPU memory and the [`hostNativeAtomicSupported`] property is 1.

Note, for objects of scopes other than `hip::thread_scope_system` this is a
  data-race, and thefore also prohibited regardless of memory characteristics.

## Implementation-Defined Behavior

For each type `T` and [`hip::thread_scope`] `S`, the value of
  `hip::atomic<T, S>::is_always_lock_free()` is as follows:

| Type `T` | [`hip::thread_scope`] `S` | `hip::atomic<T, S>::is_always_lock_free()` |
|----------|----------------------------|---------------------------------------------|
| Any      | Any                        | `sizeof(T) <= 8`                            |

## Example

```hip
#include <hip/atomic>

__global__ void example_kernel() {
  // This atomic is suitable for all threads in the system.
  hip::atomic<int, hip::thread_scope_system> a;

  // This atomic has the same type as the previous one (`a`).
  hip::atomic<int> b;

  // This atomic is suitable for all threads on the current processor (e.g. GPU).
  hip::atomic<int, hip::thread_scope_device> c;

  // This atomic is suitable for threads in the same thread block.
  hip::atomic<int, hip::thread_scope_block> d;
}
```


[`hip::thread_scope`]: ../memory_model.md

[`hip::atomic_thread_fence`]: ./atomic/atomic_thread_fence.md

[`hip::atomic::fetch_min`]: ./atomic/fetch_min.md
[`hip::atomic::fetch_max`]: ./atomic/fetch_max.md

[`hip::std::atomic`]: https://en.cppreference.com/w/cpp/atomic/atomic

[`hip::atomic`]: ./atomic.md

[atomics.types.int]: https://eel.is/c++draft/atomics.types.int
[atomics.types.pointer]: https://eel.is/c++draft/atomics.types.pointer

[`concurrentManagedAccess`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#abfea758a5672cb7803ac467543c11b67
[`hostNativeAtomicSupported`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#a64696b1aa1789ca322e8c86b69b57e7c
