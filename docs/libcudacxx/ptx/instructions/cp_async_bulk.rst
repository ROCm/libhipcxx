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

.. _libcudacxx-ptx-instructions-cp-async-bulk:

cp.async.bulk
=============

This page documents the ``cuda::ptx`` wrappers for the cp.async.bulk PTX instruction, which asynchronously copies a block of bytes between global and shared memory.

-  PTX ISA:
   `cp.async.bulk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk>`__

Implementation notes
--------------------

.. note::

   Both ``srcMem`` and ``dstMem`` must be 16-byte aligned, and ``size`` must be a multiple of 16.

Changelog
---------

-  In earlier versions, ``cp_async_bulk_multicast`` was enabled for
   SM_90. This has been changed to SM_90a.


Unicast
-------

.. include:: generated/cp_async_bulk.rst

Multicast
---------

.. include:: generated/cp_async_bulk_multicast.rst
