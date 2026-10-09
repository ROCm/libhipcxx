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
  :description: API reference for cuda::is_aligned, which checks whether a pointer is aligned to a given alignment in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, is_aligned, pointer alignment, memory, alignment

.. _libcudacxx-extended-api-memory-is_aligned:

``cuda::is_aligned``
====================

This page documents ``cuda::is_aligned``, which checks whether a pointer is aligned to the specified alignment.

Defined in the header ``<cuda/memory>``.

.. code:: cpp

   namespace cuda {

   [[nodiscard]] __host__ __device__ inline
   bool is_aligned(const void* ptr, size_t alignment) noexcept

   } // namespace cuda

The function determines if a pointer is aligned to a specific alignment.

**Parameters**

- ``ptr``: The pointer.
- ``alignment``: The alignment.

**Return value**

- ``true`` if the pointer is aligned to the specified alignment, ``false`` otherwise.

**Constraints**

- ``alignment`` must be a power of two.

.. note::

  The function is similar to the C++ standard library function `cuda::std::is_sufficiently_aligned() <https://en.cppreference.com/w/cpp/memory/is_sufficiently_aligned.html>`__ from the ``<cuda/std/memory>`` header. The differences are the following:

  - ``cuda::is_aligned()`` doesn't have a template parameter and might be less expensive to compile.
  - ``cuda::is_aligned()`` supports run-time values of ``alignment``.
  - ``cuda::std::is_sufficiently_aligned()`` additionally checks the compatibility between the alignment of the pointer type and the specified alignment.

Example
-------

.. code:: cpp

    #include <cuda/memory>

    __global__ void kernel(const void* ptr) {
        assert(cuda::is_aligned(ptr, 16));
    }

    int main() {
        void* ptr;
        hipMalloc(&ptr, 100 * sizeof(int));
        kernel<<<1, 1>>>(ptr);
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/Tr45EoKsT>`__
