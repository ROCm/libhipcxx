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
  :description: libhipcxx C++ Standard Library features
  :keywords: libhipcxx, ROCm, HIP, C++, standard library, host, device, heterogeneous

.. _libhipcxx-standard-library-features:

********************************************************************
C++ Standard Library features
********************************************************************

If you are a C++ developer, then you know the C++ Standard Library (`sometimes referred to as "The
STL" <https://stackoverflow.com/questions/5205491/whats-the-difference-between-stl-and-c-standard-library>`_)
as what comes along with your compiler and provides things like ``std::string``, ``std::vector``, or
``std::atomic``. It provides the fundamental abstractions that C++ developers need to build high
quality applications and libraries.

By default, these abstractions aren't available when writing HIP C++ device code because they don't
have the necessary ``__host__ __device__`` decorators, and their implementation may not be suitable
for use in and across host and device code.

libhipcxx solves this problem by providing an opt-in, incremental, heterogeneous implementation of
C++ Standard Library features:

1. **Opt-in**: It does not replace the Standard Library provided by your host compiler, meaning
   anything in ``std::``.
2. **Incremental**: It does not provide a complete C++ Standard Library implementation.
3. **Heterogeneous**: It works in both host and device code, as well as passing between host and
   device code.

If you know how to use headers such as ``<atomic>`` or ``<type_traits>`` from the C++ Standard
Library, then you know how to use libhipcxx. Add ``cuda/std/`` to the start of your includes and
``cuda::`` before any uses of ``std::``:

.. code-block:: cpp

    #include <cuda/std/atomic>

    cuda::std::atomic<int> x;

.. note::

   libhipcxx does not provide its own documentation for Standard Library features. Instead,
   libhipcxx :ref:`documents which Standard Library headers <libcudacxx-standard-api>` are made
   available, and defers documentation of individual features within those headers to other sources
   such as `cppreference <https://en.cppreference.com/w/>`_.
