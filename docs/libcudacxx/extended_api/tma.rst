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
  :description: Overview of the libhipcxx Tensor Memory Accelerator (TMA) extended API, which is not supported in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, TMA, tensor memory accelerator, make_tma_descriptor, DLPack

.. _libcudacxx-extended-api-tma:

Tensor Memory Accelerator (TMA)
===============================

This page covers the Tensor Memory Accelerator (TMA) extended API.

The Tensor Memory Accelerator API is not supported in libhipcxx. See :ref:`libhipcxx-limitations`.

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst. The pages are
   also listed in exclude_patterns in docs/conf.py. TMA is an NVIDIA Hopper hardware feature.

   .. toctree::
      :hidden:
      :maxdepth: 1

      cuda::make_tma_descriptor <tma/make_tma_descriptor>

   .. list-table::
      :widths: 25 45 30
      :header-rows: 1

      * - **Header**
        - **Content**
        - **libhipcxx Availability**

      * - :ref:`make_tma_descriptor() <libcudacxx-extended-api-tma-make_tma_descriptor>`
        - Construct a TMA descriptor from a DLPack tensor
        - libhipcxx 3.4
