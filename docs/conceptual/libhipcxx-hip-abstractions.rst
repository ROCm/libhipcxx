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
  :description: Understand the libhipcxx namespace hierarchy, including cuda::std::, cuda::, cuda::device::, and their hip:: aliases, and how each tier maps to host and device code.
  :keywords: libhipcxx, ROCm, HIP, namespaces, hip aliasing, cuda, std, device, AMD GPU, heterogeneous

.. _libhipcxx-hip-abstractions:

********************************
Namespace hierarchy in libhipcxx
********************************

libhipcxx organizes its APIs into three tiers, each reflecting where code can run and how closely
it conforms to the C++ Standard. Understanding this hierarchy tells you which header to include
and which namespace to use for any given facility.

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Namespace
     - Include path
     - Where it runs
   * - ``std::``
     - ``<*>``
     - ``__host__`` only. Your host compiler's Standard Library. libhipcxx does not replace or
       interfere with it.
   * - ``cuda::std::`` / ``hip::std::``
     - ``<cuda/std/*>`` / ``<hip/std/*>``
     - ``__host__`` and ``__device__``. Conforming implementations of Standard Library facilities.
   * - ``cuda::`` / ``hip::``
     - ``<cuda/*>`` / ``<hip/*>``
     - ``__host__`` and ``__device__``. Extensions to the Standard Library with GPU-specific
       semantics.
   * - ``cuda::device::`` / ``hip::device::``
     - ``<cuda/warp>``, ``<cuda/work_stealing>`` (and their ``hip/`` equivalents)
     - ``__device__`` only. Extensions that rely on GPU-only hardware features such as warp
       intrinsics.

The ``cuda::`` and ``hip::`` prefixes, and their include path equivalents, are interchangeable
aliases that resolve to the same headers. Use whichever fits your project's conventions.

The following example shows all three libhipcxx tiers side by side:

.. code-block:: cpp

    // Standard C++, __host__ only.
    #include <atomic>
    std::atomic<int> x;

    // HIP C++, __host__ __device__.
    // Strictly conforming to the C++ Standard.
    #include <cuda/std/atomic>
    cuda::std::atomic<int> x;

    // HIP C++, __host__ __device__.
    // Conforming extensions to the C++ Standard.
    #include <cuda/atomic>
    cuda::atomic<int, cuda::thread_scope_block> x;
