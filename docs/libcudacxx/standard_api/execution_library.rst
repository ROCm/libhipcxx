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


.. meta::
  :description: Documents the execution library header <cuda/std/execution> in libhipcxx, which provides execution environment properties and queries for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, execution, env, prop, get_env

.. _libcudacxx-standard-api-execution:

Execution Library
=================

This page documents ``<cuda/std/execution>``, which provides the execution environment facilities
``cuda::std::execution::prop``, ``cuda::std::execution::env`` and ``cuda::std::execution::get_env``.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/execution\> <https://en.cppreference.com/w/cpp/header/execution>`_
     - Fundamental library concepts
     - libhipcxx 3.0


Omissions
---------

-  At present, only the following features are implemented:

   -  `cuda::std::execution::prop <https://eel.is/c++draft/exec.prop>`_
   -  `cuda::std::execution::env <https://eel.is/c++draft/exec.env>`_
   -  `cuda::std::execution::get_env <https://eel.is/c++draft/exec.get.env>`_
