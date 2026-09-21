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
  :description: libhipcxx C++ Standard Library extensions
  :keywords: libhipcxx, ROCm, HIP, C++, extensions, thread scope, atomic

.. _libhipcxx-extensions:

********************************************************************
C++ Standard Library extensions
********************************************************************

libhipcxx provides HIP C++ developers with familiar Standard Library utilities to improve
productivity and flatten the learning curve of HIP. However, there are many aspects of writing
high-performance HIP C++ code that cannot be expressed through purely Standard conforming APIs. For
these cases, libhipcxx also provides *extensions* of Standard Library utilities.

For example, libhipcxx extends ``atomic<T>`` and other synchronization primitives with the notion of
a thread scope, which controls the strength of the memory fence.

To use utilities that are extensions to Standard Library features, drop the ``std``:

.. code-block:: cpp

    #include <cuda/atomic>

    cuda::atomic<int, cuda::thread_scope_device> x;

See the :ref:`Extended API <libcudacxx-extended-api>` section for the full list of extensions.
