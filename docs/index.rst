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
  :description: libhipcxx documentation
  :keywords: libhipcxx, ROCm, HIP, C++, standard library, heterogeneous, documentation

.. _index:

******************************************
libhipcxx documentation
******************************************

libhipcxx is the C++ Standard Library for HIP. It provides an opt-in, incremental, heterogeneous
implementation of C++ Standard Library features that work in both host and device code, along with
extensions to those features and abstractions that are fundamental to the HIP C++ programming model.

libhipcxx is derived from `libcudacxx <https://github.com/NVIDIA/cccl>`_ and aims to support the same
APIs on AMD GPUs. It is a header-only library, so there is nothing to compile or link against: you
only need the headers on your include path. There is no CUDA backend for libhipcxx.

The libhipcxx public repository is located at `https://github.com/ROCm/libhipcxx <https://github.com/ROCm/libhipcxx>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install libhipcxx <install/install>`
    * :doc:`Build from source <install/source-build>`

  .. grid-item-card:: Conceptual

    * :doc:`Standard library features <conceptual/libhipcxx-standard-library-features>`
    * :doc:`Standard library extensions <conceptual/libhipcxx-extensions>`
    * :doc:`HIP-specific abstractions and namespaces <conceptual/libhipcxx-hip-abstractions>`

  .. grid-item-card:: How to

    * :doc:`Add libhipcxx to a CMake project <how-to/use-libhipcxx-in-a-project>`
    * :doc:`Run the libhipcxx tests <how-to/run-libhipcxx-tests>`

  .. grid-item-card:: API reference

    * :doc:`Standard API <libcudacxx/standard_api>`
    * :doc:`Extended API <libcudacxx/extended_api>`
    * :doc:`PTX API <libcudacxx/ptx>`
    * :ref:`libhipcxx-limitations`
    * :ref:`libhipcxx-conformance`
    * :ref:`genindex`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
