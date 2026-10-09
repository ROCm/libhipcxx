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
  :description: API reference for cuda::align_up, which rounds a pointer up to the closest address aligned to a given alignment in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, align_up, pointer alignment, memory, alignment

.. _libcudacxx-extended-api-memory-align_up:

``cuda::align_up``
==================

This page documents ``cuda::align_up``, which aligns a pointer up to the specified alignment.

Defined in the header ``<cuda/memory>``.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ inline
   T* align_up(T* ptr, size_t alignment) noexcept;

   } // namespace cuda

The function returns the original pointer or closest pointer larger than ``ptr`` that is aligned to the specified alignment :math:`ceil\left(\frac{ptr}{alignment}\right) * alignment`.

**Parameters**

- ``ptr``: The pointer.
- ``alignment``: The alignment.

**Return value**

- The original pointer or closest pointer larger than ``ptr`` that is aligned to the specified alignment.

**Constraints**

- ``alignment`` must be a power of two.
- ``alignment >= alignof(T)``.
- ``ptr`` is aligned to ``alignof(T)``.

**Performance considerations**

- The function is optimized for compile-time values of ``alignment``.
- The function does not perform any operations if ``alignment == alignof(T)``.
- The returned pointer is decorated with ``__builtin_assume_aligned`` to help the compiler generate better code.
- The returned pointer maintains the same memory space, for example shared memory, as the input pointer.

..
   - The function is translated to ``LOP3.LUT`` + ``IADD.64`` instructions for other values of ``alignment``.

Example
-------

.. code:: cpp

    #include <cuda/memory>

    __global__ void kernel(int* ptr) {
        auto ptr_align16 = cuda::align_up(ptr, 16);
        reinterpret_cast<int4*>(ptr_align16)[0] = int4{1, 2, 3, 4};
    }

    int main() {
        int* ptr;
        hipMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/d8e5KETeE>`__
