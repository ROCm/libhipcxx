..
    MIT License

    Copyright (c) 2024-2026 Advanced Micro Devices, Inc.

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
  :description: API reference for cuda::associate_access_property, which associates a cache access property with a raw pointer for subsequent memory operations in libhipcxx.
  :keywords: libhipcxx, ROCm, HIP, C++, associate_access_property, access property, pointer, cache, streaming, global memory

.. _libcudacxx-extended-api-memory-access-properties-associate-access-property:

``cuda::associate_access_property``
===================================

This page documents ``cuda::associate_access_property``, which associates a cache access property with a raw pointer for use in subsequent memory operations.

.. code:: cuda

   template <class T, class Property>
   __host__ __device__
   T* associate_access_property(T* ptr, Property prop);

**Preconditions**:

- if ``Property`` is :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`
  then it must be valid to cast the generic pointer ``ptr`` to a pointer to the shared memory address space.
- if ``Property`` is one of :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-global>`,
  :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`,
  :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-normal>`, or
  :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-streaming>`
  then it must be valid to cast the generic pointer ``ptr`` to a pointer to the global memory address space.
- if ``Property`` is a :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`
  of "range" kind, then ``ptr`` must be in the valid range.

**Mandates**: ``Property`` is convertible to :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`.

**Effects**: no effects.

**Hint**: to associate an access property with the returned pointer, such that subsequent memory operations with the
returned pointer *or* pointers derived from it *may* apply the access property.

-  The "association" is *not* part of the value representation of the pointer.
-  The compiler is allowed to drop the association; it does not have a functional consequence.
-  The association *may* hold through simple expressions, sequence of simple statements, or fully inlined function
   calls where the pointer value or C++ reference is provably unchanged; this includes offset pointers used for
   array access.
-  The association is *not* expected to hold through the ABI of an unknown function call, e.g., when the pointer is
   passed through a separately-compiled function interface, unless link-time optimizations are used.

**Note**: currently ``associate_access_property`` is ignored by nvcc and nvc++ on the host; but this might change any time.

Example
-------

.. code:: cuda

   #include <cuda/cooperative_groups.h>
   __global__ void memcpy(int const* in_, int* out) {
       int const* in = cuda::associate_access_property(in_, cuda::access_property::streaming{});
       auto idx = cooperative_groups::this_grid().thread_rank();

       __shared__ int shmem[N];
       shmem[threadIdx.x] = in[idx]; // streaming access

       // compute...
   }
