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
  :description: Overview of the convenience macros in libhipcxx for HIP that detect system and compile-time properties via the preprocessor.
  :keywords: libhipcxx, ROCm, HIP, C++, macros, preprocessor, operating system detection, compile-time properties

.. _libcudacxx-extended-api-macros:

======
Macros
======

libhipcxx provides a set of convenience macros for detecting various system and compile-time
properties via the preprocessor. These macros are available when any libhipcxx header is
included, and do not require including a specific header file.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Macro**
     - **Content**
     - **Since**

   * - ``_CCCL_OS(<os>)``, for example ``_CCCL_OS(LINUX)`` or ``_CCCL_OS(WINDOWS)``
     - Detecting the current operating system.
     - libhipcxx 3.4
