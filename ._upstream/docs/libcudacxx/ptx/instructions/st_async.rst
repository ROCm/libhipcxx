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

.. _libcudacxx-ptx-instructions-st-async:

st.async
========

This page documents the ``cuda::ptx`` wrappers for the st.async PTX instruction, which performs an asynchronous store to a remote cluster memory address and signals an mbarrier on completion.

-  PTX ISA:
   `st.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async>`__
-  Used in: :ref:`How to use st.async <libcudacxx-ptx-examples-st-async>`

.. note::

   Alignment of ``addr`` must be a multiple of vector size. For instance, the ``addr`` supplied to the ``v2.b32`` variant must be aligned to ``2 x 4 = 8`` bytes.

.. include:: generated/st_async.rst
