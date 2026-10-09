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
  :description: API reference for cuda::aligned_size_t, a shape type representing a byte extent with a statically defined address and size alignment for memory operations in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, aligned_size_t, alignment, byte extent, memcpy_async, shape

.. _libcudacxx-extended-api-memory-aligned-size:

``cuda::aligned_size_t``
========================

This page documents ``cuda::aligned_size_t``, a shape type representing a byte extent with a statically defined address and size alignment for memory operations.

Defined in headers ``<cuda/memory>``, ``<cuda/barrier>`` and ``<cuda/pipeline>``:

.. code:: cpp

   template <cuda::std::size_t Alignment>
   struct cuda::aligned_size_t {
     static constexpr cuda::std::size_t align = Align;
     cuda::std::size_t value;
     __host__ __device__ explicit constexpr aligned_size(cuda::std::size_t size);
     __host__ __device__ constexpr operator cuda::std::size_t();
   };

The class template ``cuda::aligned_size_t`` is a *shape* representing an extent of bytes with a statically
defined (address and size) alignment.

*Preconditions*:

-  The *address* of the extent of bytes must be aligned to an ``Alignment`` alignment boundary.
-  The *size* of the extent of bytes must be a multiple of the ``Alignment``.

Template Parameters
-------------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Alignment``
     - The address and size alignment of the byte extent.

Data Members
------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``align``
     - The alignment of the byte extent.
   * - ``value``
     - The size of the byte extent.

Member Functions
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``(constructor)``
     - Constructs an *aligned size*. If the ``size`` is not a multiple of ``Alignment`` the behavior is undefined.
   * - ``(destructor)``
     - Trivial implicit destructor.
   * - ``operator=``
     - Trivial implicit copy/move.
   * - ``operator cuda::std::size_t``
     - Implicit conversion to `cuda::std::size_t <https://en.cppreference.com/w/cpp/types/size_t>`__.

Notes
-----

If ``Alignment`` is not a `valid alignment <https://en.cppreference.com/w/c/language/object#Alignment>`_,
the behavior is undefined.

Example
-------

.. code:: cpp

   #include <cuda/memory>

   __global__ void example_kernel(void* dst, void* src, size_t size) {
     cuda::barrier<cuda::thread_scope_system> bar;
     init(&bar, 1);

     // Implementation cannot make assumptions about alignment.
     cuda::memcpy_async(dst, src, size, bar);

     // Implementation can assume that dst and src are 16-bytes aligned,
     // and that size is a multiple of 16, and may optimize accordingly.
     cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(size), bar);

     bar.arrive_and_wait();
   }

..
   `See it on Godbolt <https://godbolt.org/z/PWGdfTd7d>`_
