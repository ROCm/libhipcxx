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

.. _libcudacxx-ptx-instructions-mbarrier-try_wait:

mbarrier.try_wait
=================

This page documents the ``cuda::ptx`` wrappers for the mbarrier.try_wait PTX instruction, which attempts to wait for a shared memory barrier to be satisfied, with a timeout.

-  PTX ISA:
   `mbarrier.try_wait <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait>`__


.. _mbarrier.try_wait-1:

mbarrier.try_wait
-----------------

.. include:: generated/mbarrier_try_wait.rst

mbarrier.try_wait.parity
------------------------

.. include:: generated/mbarrier_try_wait_parity.rst
