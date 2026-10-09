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
  :description: Explore libhipcxx platform limitations and unsupported APIs, including which libcudacxx APIs are not available for AMD GPU development with HIP.
  :keywords: libhipcxx, ROCm, HIP, limitations, unsupported, APIs, libcudacxx, AMD GPU, CUDA

.. _libhipcxx-limitations:

********************************************************************
Limitations and unsupported APIs
********************************************************************

Platform limitations
====================

libhipcxx has the following platform limitations.

* libhipcxx does not support the CUDA backend or NVIDIA hardware.
* ``cuda::std::chrono::system_clock::now()`` does not return a UNIX timestamp. The host system clock
  and the device system clock are not synchronized and may run at different clock rates.

.. important::

   Some APIs from libcudacxx may be missing or may not achieve the same performance on AMD GPUs
   due to fundamental hardware architecture differences between AMD and NVIDIA GPUs. APIs that rely
   on NVIDIA-specific hardware features (such as PTX instructions, tensor cores, or specific
   compute capabilities) may be unavailable, provide degraded functionality, or fall back to
   software implementations with reduced performance.

Unsupported APIs
================

The following APIs from libcudacxx are **not** supported in libhipcxx:

.. list-table::
  :widths: 30 20 50
  :header-rows: 1

  * - Group
    - API
    - Description
  * - Synchronization Library
    - ``<cuda/std/latch>``
    - Single-phase asynchronous thread-coordination mechanism.
  * - Synchronization Library
    - ``<cuda/std/barrier>``
    - Multi-phase asynchronous thread-coordination mechanism.
  * - Synchronization Library
    - ``<cuda/std/semaphore>``
    - Primitives for constraining concurrent access.
  * - Extended Synchronization Library
    - ``<cuda/latch>``
    - System-wide ``cuda::std::latch`` single-phase asynchronous thread coordination mechanism.
  * - Extended Synchronization Library
    - ``<cuda/barrier>``
    - System-wide ``cuda::std::barrier`` multi-phase asynchronous thread coordination mechanism.
  * - Extended Synchronization Library
    - ``<cuda/semaphore>``
    - System-wide primitives for constraining concurrent access.
  * - Extended Synchronization Library
    - ``<cuda/pipeline>``
    - Coordination mechanisms to sequence asynchronous operations.
  * - Extended Memory Access Properties Library
    - ``<cuda/annotated_ptr>``
    - Memory access properties for pointers.
  * - Extended Memory Access Properties Library
    - ``<cuda/discard_memory>``
    - Discards modified cache lines without writing them back. The header is available, but
      ``cuda::discard_memory`` has no effect on AMD GPUs.
  * - Extended Warp Library
    - ``<cuda/warp>``
    - Warp shuffle, match and lane-mask functions. The header is available, but
      ``cuda::device::warp_shuffle_*``, ``cuda::device::warp_match_all`` and ``cuda::device::lane_mask``
      are only compiled for NVIDIA PTX targets.
  * - Extended Work Stealing Library
    - ``<cuda/work_stealing>``
    - ``cuda::for_each_canceled_block`` for cancelling and stealing thread blocks. On AMD GPUs it
      invokes the function once for the current block and does not cancel other blocks.
  * - Device-Level APIs
    - ``cuda::device::*``
    - Hardware-specific device functions including warp shuffles, barrier operations with
      transaction counts, and async memory operations. These require NVIDIA PTX instructions
      and SM-specific hardware features not available on AMD GPUs.
  * - Extended TMA Library
    - ``<cuda/tma>``
    - ``cuda::make_tma_descriptor`` creates descriptors for the NVIDIA Tensor Memory Accelerator.
  * - Parallel Algorithms
    - ``<cuda/std/algorithm>``, ``<cuda/std/numeric>``
    - Overloads of the standard algorithms that take an execution policy such as
      ``cuda::execution::gpu``.
  * - PTX API
    - ``<cuda/ptx>``
    - The ``cuda::ptx`` namespace contains functions that map to NVIDIA PTX instructions.
  * - CUDA Tile
    - Tile mode
    - Support for compiling libcudacxx headers in NVIDIA CUDA Tile mode.
  * - Runtime API
    - ``<cuda/buffer>``
    - ``cuda::buffer`` and the other stream-ordered containers.
  * - Runtime API
    - ``cuda::device::current_arch_id``, ``cuda::device::current_arch_traits``,
      ``cuda::device::current_compute_capability``
    - Queries for the architecture of the current device.
