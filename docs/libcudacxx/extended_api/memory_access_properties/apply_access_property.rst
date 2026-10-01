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
  :description: API reference for cuda::apply_access_property, which prefetches memory while applying a cache residence property in libhipcxx for HIP and CUDA.
  :keywords: libhipcxx, ROCm, HIP, C++, apply_access_property, prefetch, cache, persisting, normal, global memory

.. _libcudacxx-extended-api-memory-access-properties-apply-access-property:

``cuda::apply_access_property``
===============================

This page documents ``cuda::apply_access_property``, which prefetches a memory range while applying a cache residence property.

.. code:: cuda

   template <class ShapeT>
   __host__ __device__
   void apply_access_property(void const volatile* ptr, ShapeT shape, cuda::access_property::persisting) noexcept;
   template <class ShapeT>
   __host__ __device__
   void apply_access_property(void const volatile* ptr, ShapeT shape, cuda::access_property::normal) noexcept;

**Mandates**: :ref:`ShapeT <libcudacxx-extended-api-memory-access-shapes>` is either `std::size_t <https://en.cppreference.com/w/cpp/types/size_t>`_ or
:ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-access-shapes-aligned-size>`.

**Preconditions**: ``ptr`` points to a valid allocation for ``shape`` in the global memory address space.

**Effects**: no effects.

**Hint**: to prefetch ``shape`` bytes of memory starting at ``ptr`` while applying a property. Two properties are supported:

-  :ref:`cuda::access_property::persisting <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`
-  :ref:`cuda::access_property::normal <libcudacxx-extended-api-memory-access-properties-access-property-normal>`

**Note**: In **Preconditions**, "valid allocation for ``shape``" means
that:

-  if ``ShapeT`` is ``aligned_size_t<N>(sz)`` then ``ptr`` is aligned to an ``N``-bytes alignment boundary, and
-  for all offsets ``i`` in the extent of ``shape``, i.e., ``i`` in ``[0, shape)`` then the expression ``*(ptr + i)``
   does not exhibit undefined behavior.

**Note**: currently ``apply_access_property`` is ignored by nvcc and nvc++ on the host.

Example
-------

Given three input and output vectors ``x``, ``y``, and ``z``, and two arrays of coefficients ``a`` and ``b``,
all of length ``N``:

.. code:: cuda

   size_t N;
   int* x, *y, *z;
   int* a, *b;

the grid-strided kernel:

.. code:: cuda

   __global__ void update(int* const x, int const* const a, int const* const b, size_t N) {
       auto g = cooperative_groups::this_grid();
       for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
           x[idx] = a[idx] * x[idx] + b[idx];
       }
   }

updates ``x``, ``y``, and ``z`` as follows:

.. code:: cuda

   update<<<grid, block>>>(x, a, b, N);
   update<<<grid, block>>>(y, a, b, N);
   update<<<grid, block>>>(z, a, b, N);

The elements of ``a`` and ``b`` are used in all kernels. For certain values of ``N``, this may prevent parts of ``a``
and ``b`` from being evicted from the L2 cache, avoiding reloading these from memory in the subsequent ``update`` kernel.

With :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>` and
:ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`, we can
write kernels that specify that ``a`` and ``b`` are accessed more often than (``pin``) and as often as (``unpin``) other data:

.. code:: cuda

   __global__ void pin(int* a, int* b, size_t N) {
       auto g = cooperative_groups::this_grid();
       for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
           cuda::apply_access_property(a + idx, sizeof(int), cuda::access_property::persisting{});
           cuda::apply_access_property(b + idx, sizeof(int), cuda::access_property::persisting{});
       }
   }
   __global__ void unpin(int* a, int* b, size_t N) {
       auto g = cooperative_groups::this_grid();
       for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
           cuda::apply_access_property(a + idx, sizeof(int), cuda::access_property::normal{});
           cuda::apply_access_property(b + idx, sizeof(int), cuda::access_property::normal{});
       }
   }

which we can launch before and after the ``update`` kernels:

.. code:: cuda

   pin<<<grid, block>>>(a, b, N);
   update<<<grid, block>>>(x, a, b, N);
   update<<<grid, block>>>(y, a, b, N);
   update<<<grid, block>>>(z, a, b, N);
   unpin<<<grid, block>>>(a, b, N);

This does not require modifying the ``update`` kernel, and for certain values of ``N`` prevents ``a`` and ``b``
from having to be re-loaded from memory.

The ``pin`` and ``unpin`` kernels can be fused into the kernels for the ``x`` and ``z`` updates by modifying these kernels.
