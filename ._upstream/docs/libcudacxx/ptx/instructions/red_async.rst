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

.. _libcudacxx-ptx-instructions-mbarrier-red-async:

red.async
=========

This page documents the ``cuda::ptx`` wrappers for the red.async PTX instruction, which performs an asynchronous reduction operation on shared or global memory.

-  PTX ISA:
   `red.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red-async>`__

.. _red.async-1:

red.async
---------

.. include:: generated/red_async.rst

red.async ``.s64`` emulation
----------------------------

PTX does not currently (CTK 12.3) expose ``red.async.add.s64``. This
exposure is emulated in ``cuda::ptx`` using

.. code:: cuda

   // red.async.relaxed.cluster.shared::cluster.mbarrier::complete_tx::bytes{.op}.u64  [dest], value, [remote_bar]; // .u64 intentional PTX ISA 81, SM_90
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void red_async(
     cuda::ptx::op_add_t,
     int64_t* dest,
     const int64_t& value,
     int64_t* remote_bar);
