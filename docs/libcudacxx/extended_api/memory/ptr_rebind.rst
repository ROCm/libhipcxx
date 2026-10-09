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
  :description: API reference for cuda::ptr_rebind, which casts a pointer to a pointer of a different type as a safer alternative to reinterpret_cast in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, ptr_rebind, pointer cast, reinterpret_cast, alignment, memory

.. _libcudacxx-extended-api-memory-ptr_rebind:

``cuda::ptr_rebind``
====================

This page documents ``cuda::ptr_rebind``, which rebinds a pointer to a different type.

Defined in the header ``<cuda/memory>``.

.. code:: cpp

    namespace cuda {

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    U* ptr_rebind(T* ptr) noexcept;

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    const U* ptr_rebind(const T* ptr) noexcept;

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    volatile U* ptr_rebind(volatile T* ptr) noexcept;

    template <typename U, typename T>
    [[nodiscard]] __host__ __device__
    const volatile U* ptr_rebind(const volatile T* ptr) noexcept;

    } // namespace cuda

The functions return the pointer ``ptr`` cast to type ``U*`` or ``const U*``. They are shorter and safer alternative to ``reinterpret_cast``.

**Parameters**

- ``ptr``: The pointer.

**Return value**

- The pointer cast to type ``U*`` or ``const U*``.

**Constraints**

- ``ptr`` must be aligned to ``alignof(U)`` and ``alignof(T)``.

**Performance considerations**

- The returned pointer is decorated with ``__builtin_assume_aligned`` with the ``alignof(U)`` value to help the compiler generate better code.
- The returned pointer maintains the same memory space, for example shared memory, as the input pointer.

Example
-------

.. code:: cpp

    #include <cuda/memory>
    #include <cuda/std/cstdint>

    __global__ void kernel(const int* ptr, volatile int* ptr2) {
        auto ptr_res1 = cuda::ptr_rebind<uint64_t>(ptr);  // ptr_res1: const uint64_t*
        auto ptr_res2 = cuda::ptr_rebind<uint64_t>(ptr2); // ptr_res2: volatile uint64_t*
    }

    int main() {
        int* ptr;
        hipMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/bavzabce9>`__
