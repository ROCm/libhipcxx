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
  :description: libhipcxx limitations and unsupported APIs
  :keywords: libhipcxx, ROCm, HIP, limitations, unsupported, APIs

.. _libhipcxx-limitations:

********************************************************************
Limitations and unsupported APIs
********************************************************************

Platform limitations
====================

* libhipcxx does not support the CUDA backend or NVIDIA hardware.
* libhipcxx does not support the Windows operating system.
* ``cuda::std::chrono::system_clock::now()`` does not return a UNIX timestamp. The host system clock
  and the device system clock are not synchronized and may run at different clock rates.

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
  * - PTX API
    - ``<cuda/ptx>``
    - The ``cuda::ptx`` namespace contains functions that map to NVIDIA PTX instructions.
