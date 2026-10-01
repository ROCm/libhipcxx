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

.. _libcudacxx-ptx-instructions-cp-reduce-async-bulk:

cp.reduce.async.bulk
====================

This page documents the ``cuda::ptx`` wrappers for the cp.reduce.async.bulk PTX instruction, which asynchronously performs a reduction operation and stores the result to a destination.

-  PTX ISA:
   `cp.reduce.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-reduce-async-bulk>`__


Integer and floating point instructions
---------------------------------------

.. include:: generated/cp_reduce_async_bulk.rst

Emulation of ``.s64`` instruction
---------------------------------

PTX does not currently (CTK 12.3) expose
``cp.reduce.async.bulk.add.s64``. This exposure is emulated in
``cuda::ptx`` using:

.. code:: cuda

   // cp.reduce.async.bulk.dst.src.mbarrier::complete_tx::bytes.op.u64 [dstMem], [srcMem], size, [rdsmem_bar]; // 2. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .shared::cta }
   // .type      = { .s64 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_add_t,
     int64_t* dstMem,
     const int64_t* srcMem,
     uint32_t size,
     uint64_t* rdsmem_bar);

   // cp.reduce.async.bulk.dst.src.bulk_group.op.u64  [dstMem], [srcMem], size; // 6. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   // .type      = { .s64 }
   // .op        = { .add }
   template <typename=void>
   __device__ static inline void cp_reduce_async_bulk(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     cuda::ptx::op_add_t,
     int64_t* dstMem,
     const int64_t* srcMem,
     uint32_t size);

FP16 instructions
-----------------

.. include:: generated/cp_reduce_async_bulk_f16.rst

BF16 instructions
-----------------

.. include:: generated/cp_reduce_async_bulk_bf16.rst
