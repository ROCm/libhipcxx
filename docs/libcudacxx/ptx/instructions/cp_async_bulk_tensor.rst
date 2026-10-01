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

.. _libcudacxx-ptx-instructions-cp-async-bulk-tensor:

cp.async.bulk.tensor
====================

This page documents the ``cuda::ptx`` wrappers for the cp.async.bulk.tensor PTX instruction, which asynchronously copies tensor data between global and shared memory using a tensor map.

-  PTX ISA:
   `cp.async.bulk.tensor <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor>`__

Changelog
---------

-  In earlier versions, ``cp_async_bulk_tensor_multicast`` was enabled
   for SM_90. This has been changed to SM_90a.

Unicast
-------

.. include:: generated/cp_async_bulk_tensor.rst

Multicast
---------

.. include:: generated/cp_async_bulk_tensor_multicast.rst

Scatter / Gather
----------------

.. include:: generated/cp_async_bulk_tensor_gather_scatter.rst
