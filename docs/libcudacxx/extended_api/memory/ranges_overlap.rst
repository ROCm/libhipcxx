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
  :description: API reference for cuda::ranges_overlap, which checks whether two half-open ranges intersect in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, ranges_overlap, range intersection, iterators, memory

.. _libcudacxx-extended-api-memory-ranges_overlap:

``cuda::ranges_overlap``
========================

This page documents ``cuda::ranges_overlap``, which checks whether two ranges overlap.

Defined in the ``<cuda/memory>`` header.

.. code:: cpp

   namespace cuda {

   template <typename T>
   [[nodiscard]] __host__ __device__ constexpr
   bool ranges_overlap(T lhs_start, T lhs_end, T rhs_start, T rhs_end) noexcept;

   } // namespace cuda

Returns ``true`` when the half-open byte ranges ``[lhs_start, lhs_end)`` and ``[rhs_start, rhs_end)`` intersect.

**Constraints**

- ``T`` must be a forward iterator.

**Parameters**

- ``lhs_start``: The beginning of the first range.
- ``lhs_end``: The end of the first range.
- ``rhs_start``: The beginning of the second range.
- ``rhs_end``: The end of the second range.

**Return value**

- ``true`` when the two ranges overlap, ``false`` otherwise.

**Performance considerations**

- The function is optimized when the ranges are contiguous and random access iterators.

Example
-------

.. code:: cpp

    #include <cuda/memory>
    #include <cuda/std/cassert>

    __global__ void overlap_kernel() {
        int arrayA[10];
        int arrayB[10];
        assert(cuda::ranges_overlap(arrayA + 2, arrayA + 7, arrayA, arrayA + 10)); // overlap
        assert(!cuda::ranges_overlap(arrayA, arrayA + 10, arrayB, arrayB + 10));   // no overlap
    }

    int main() {
        overlap_kernel<<<1, 1>>>();
        hipDeviceSynchronize();
        return 0;
    }

..
   `See it on Godbolt 🔗 <https://godbolt.org/z/nasnWz9Tv>`__
