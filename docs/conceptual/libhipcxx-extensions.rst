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
  :description: Learn about libhipcxx extensions to the C++ Standard Library.
  :keywords: libhipcxx, ROCm, HIP, C++, extensions, thread scope, atomic, synchronization, CUDA, AMD GPU

.. _libhipcxx-extensions:

********************************************
C++ Standard Library extensions in libhipcxx
********************************************

libhipcxx provides HIP C++ developers with familiar Standard Library utilities to improve
productivity and flatten the learning curve of HIP. However, there are many aspects of writing
high-performance HIP C++ code that cannot be expressed through purely Standard conforming APIs. For
these cases, libhipcxx also provides *extensions* of Standard Library utilities.

To use utilities that are extensions to Standard Library features, drop the ``std`` from both the
include path and namespace:

.. code-block:: cpp

    #include <cuda/atomic>

    cuda::atomic<int, cuda::thread_scope_device> x;

For a full explanation of the namespace hierarchy, see
:ref:`HIP-specific abstractions and namespaces <libhipcxx-hip-abstractions>`.

Available extensions
====================

libhipcxx provides extensions across synchronization, memory, math, and device-specific
functionality. The following table lists each extension category with its key APIs:

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Extension
     - APIs
     - Notes
   * - Thread-scope synchronization
     - * ``cuda::atomic`` / ``hip::atomic``
       * ``cuda::atomic_ref`` / ``hip::atomic_ref``
     - ``cuda::thread_scope`` controls memory fence strength across thread, block, device, or system
   * - Functional utilities
     - * ``cuda::maximum`` / ``hip::maximum``
       * ``cuda::minimum`` / ``hip::minimum``
       * ``cuda::proclaim_return_type`` / ``hip::proclaim_return_type``
       * ``cuda::get_device_address`` / ``hip::get_device_address``
     -
   * - Math utilities
     - * ``cuda::ceil_div`` / ``hip::ceil_div``
       * ``cuda::round_up`` / ``hip::round_up``
       * ``cuda::round_down`` / ``hip::round_down``
       * ``cuda::ilog2`` / ``hip::ilog2``
       * ``cuda::ilog10`` / ``hip::ilog10``
     -
   * - Bit utilities
     - * ``cuda::bitmask`` / ``hip::bitmask``
       * ``cuda::bit_reverse`` / ``hip::bit_reverse``
       * ``cuda::bitfield_insert`` / ``hip::bitfield_insert``
       * ``cuda::bitfield_extract`` / ``hip::bitfield_extract``
     -
   * - Stream reference
     - * ``cuda::stream_ref`` / ``hip::stream_ref``
     - Type-safe wrapper around ``hipStream_t``
   * - Memory resources
     - * ``cuda::mr::resource`` / ``hip::mr::resource``
       * ``cuda::mr::resource_ref`` / ``hip::mr::resource_ref``
     - Experimental. Requires ``LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE``.

For per-API documentation, see the :ref:`Extended API reference <libcudacxx-extended-api>`.

Unsupported extensions
======================

Several extensions from the upstream libcudacxx project are not supported in libhipcxx because they
depend on NVIDIA hardware. These include:

- ``<cuda/latch>``, ``<cuda/barrier>``, ``<cuda/semaphore>``, ``<cuda/pipeline>`` — scoped
  synchronization primitives
- ``cuda::memcpy_async``, ``cuda::memcpy_async_tx``, ``cuda::aligned_size_t`` — asynchronous
  copies, which depend on ``<cuda/barrier>`` and ``<cuda/pipeline>``
- ``<cuda/annotated_ptr>`` — memory access properties for pointers
- ``cuda::discard_memory`` — available, but has no effect on AMD GPUs
- ``cuda::device::warp_shuffle_idx``, ``warp_shuffle_up``, ``warp_shuffle_down``,
  ``warp_shuffle_xor`` — warp shuffles, which are only compiled for NVIDIA PTX targets
- ``cuda::for_each_canceled_block`` — work stealing, which depends on NVIDIA hardware block
  cancellation. On AMD GPUs it invokes the function once for the current block.
- ``<cuda/ptx>`` — NVIDIA PTX instruction wrappers

For the complete list, see :ref:`Limitations and unsupported APIs <libhipcxx-limitations>`.
