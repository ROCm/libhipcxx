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
  :description: API reference for cuda::latch, a single-phase asynchronous thread coordination mechanism with thread scope support in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, latch, thread scope, synchronization, single-phase, count-down

.. _libcudacxx-extended-api-synchronization-latch:

``cuda::latch``
===============

This page documents ``cuda::latch``, a single-phase asynchronous thread coordination mechanism with thread scope support for counting down and waiting on completion.

Defined in header ``<cuda/latch>``:

.. code:: cpp

   template <cuda::thread_scope Scope>
   class cuda::latch;

The class template ``cuda::latch`` is an extended form of `cuda::std::latch <https://en.cppreference.com/w/cpp/thread/latch>`_
that takes an additional :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` argument.
It has the same interface and semantics as `cuda::std::latch <https://en.cppreference.com/w/cpp/thread/latch>`_.

Concurrency Restrictions
------------------------

An object of type ``cuda::latch`` or `cuda::std::latch <https://en.cppreference.com/w/cpp/thread/latch>`_ shall not
be accessed concurrently by CPU and GPU threads unless:

- it is in unified memory and the `concurrentManagedAccess property <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html#structhip_device_prop__t>`_
  is 1, or
- it is in CPU memory and the `hostNativeAtomicSupported property <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html#structhip_device_prop__t>`_
  is 1.

Note: for objects of scopes other than ``cuda::thread_scope_system``, this is a data race, and is therefore also prohibited
regardless of memory characteristics.

..
   NVIDIA-specific (compute capability), not applicable to AMD GPUs.

   Under CUDA Compute Capability 6 (Pascal) or prior, an object of type ``cuda::latch`` or
   `cuda::std::latch <https://en.cppreference.com/w/cpp/thread/latch>`_ may not be used.

Implementation-Defined Behavior
-------------------------------

For each :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``, the value of
``cuda::latch<S>::max()`` is as follows:

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``
     - ``cuda::latch<S>::max()``
   * - Any thread scope
     - ``cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max()``

Example
-------

.. code:: cpp

   #include <cuda/latch>

   __global__ void example_kernel() {
     // This latch is suitable for all threads in the system.
     cuda::latch<cuda::thread_scope_system> a(10);

     // This latch has the same type as the previous one (`a`).
     cuda::std::latch b(10);

     // This latch is suitable for all threads on the current processor (e.g. GPU).
     cuda::latch<cuda::thread_scope_device> c(10);

     // This latch is suitable for all threads in the same thread block.
     cuda::latch<cuda::thread_scope_block> d(10);
   }

..
   `See it on Godbolt <https://godbolt.org/z/8v4dcK7fa>`_
