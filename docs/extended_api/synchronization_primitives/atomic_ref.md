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

# `hip::atomic_ref`

Defined in header `<hip/atomic>`:

```hip
template <typename T, hip::thread_scope Scope = hip::thread_scope_system>
class hip::atomic_ref;
```

The class template `hip::atomic_ref` is an extended form of [`hip::std::atomic_ref`]
  that takes an additional [`hip::thread_scope`] argument, defaulted to
  `hip::std::thread_scope_system`.
It has the same interface and semantics as [`hip::std::atomic_ref`], with the
  following additional operations.
This class additionally deviates from the standard by being backported to C++11.

## Limitations

`hip::atomic_ref<T>` and `hip::std::atomic_ref<T>` may only be instantiated with a T that are either 4 or 8 bytes.

No object or subobject of an object referenced by an `atomic_­ref` shall be concurrently referenced by any other `atomic_­ref` that has a different `Scope`. 

For `hip::atomic_ref<T>` and `hip::std::atomic_ref<T>` the type `T` must satisfy the following:
- `4 <= sizeof(T) <= 8`.
- `T` must not have "padding bits", i.e., 'T`'s [object representation](https://en.cppreference.com/w/cpp/language/object#Object_representation_and_value_representation) must not have bits that do not participate in it's value representation.

## Atomic Extrema Operations

| Name                       | Description                                                                               |
|----------------------------|-------------------------------------------------------------------------------------------|
| [`hip::atomic_ref::fetch_min`] | Atomically find the minimum of the stored value and a provided value. `(member function)` |
| [`hip::atomic_ref::fetch_max`] | Atomically find the maximum of the stored value and a provided value. `(member function)` |

## Concurrency Restrictions

See [`memory model`] documentation for general restrictions on atomicity.


## Implementation-Defined Behavior

For each type `T` and [`hip::thread_scope`] `S`, the value of
  `hip::atomic_ref<T, S>::is_always_lock_free()` and 
  `hip::std::atomic_ref<T>::is_always_lock_free()` is as follows:

| Type `T` | [`hip::thread_scope`] `S` | `hip::atomic_ref<T, S>::is_always_lock_free()` |
|----------|----------------------------|---------------------------------------------|
| Any      | Any                        | `sizeof(T) <= 8`                            |

## Example

```hip
#include <hip/atomic>

__global__ void example_kernel(int *gmem, int *pinned_mem) {
  // This atomic is suitable for all threads in the system.
  hip::atomic_ref<int, hip::thread_scope_system> a(pinned_mem);

  // This atomic has the same type as the previous one (`a`).
  hip::atomic_ref<int> b(pinned_mem);

  // This atomic is suitable for all threads on the current processor (e.g. GPU).
  hip::atomic_ref<int, hip::thread_scope_device> c(gmem);

  __shared__ int shared_v;
  // This atomic is suitable for threads in the same thread block.
  hip::atomic_ref<int, hip::thread_scope_block> d(&shared);
}
```



[`hip::thread_scope`]: ../thread_scopes.md
[`memory model`]: ../memory_model.md

[`hip::atomic_thread_fence`]: ./atomic/atomic_thread_fence.md

[`hip::atomic_ref::fetch_min`]: ./atomic/fetch_min.md
[`hip::atomic_ref::fetch_max`]: ./atomic/fetch_max.md

[`hip::std::atomic_ref`]: https://en.cppreference.com/w/cpp/atomic/atomic_ref

[atomics.types.int]: https://eel.is/c++draft/atomics.types.int
[atomics.types.pointer]: https://eel.is/c++draft/atomics.types.pointer

[`concurrentManagedAccess`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#abfea758a5672cb7803ac467543c11b67
[`hostNativeAtomicSupported`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#a64696b1aa1789ca322e8c86b69b57e7c
