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
  :description: Explore libhipcxx, the C++ Standard Library for HIP on AMD GPUs. Find installation guides, API reference, conceptual topics, and how-to guides.
  :keywords: libhipcxx, ROCm, HIP, C++, standard library, heterogeneous, AMD GPU, CUDA alternative, HIP C++, documentation

.. _index:

******************************************
libhipcxx documentation
******************************************

libhipcxx is the C++ Standard Library for HIP. It provides an opt-in, incremental, heterogeneous
implementation of C++ Standard Library features that work in both host and device code, along with
extensions to those features and abstractions that are fundamental to the HIP C++ programming model.

libhipcxx is derived from `libcudacxx <https://github.com/NVIDIA/cccl/tree/main/libcudacxx>`_ and aims to support the same
APIs on AMD GPUs. It is a header-only library, so there is nothing to compile or link against: you
only need the headers on your include path. There is no CUDA backend for libhipcxx.

The libhipcxx public repository is located at `ROCm/libhipcxx <https://github.com/ROCm/libhipcxx>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install libhipcxx <install/install>`
    * :doc:`Build from source <install/source-build>`

  .. grid-item-card:: Conceptual

    * :doc:`C++ standard library features <conceptual/libhipcxx-standard-library-features>`
    * :doc:`C++ standard library extensions <conceptual/libhipcxx-extensions>`
    * :doc:`Namespace hierarchy in libhipcxx <conceptual/libhipcxx-hip-abstractions>`

  .. grid-item-card:: How to

    * :doc:`Run libhipcxx tests <how-to/run-libhipcxx-tests>`

  .. grid-item-card:: Examples

    * `libhipcxx examples <https://github.com/ROCm/rocm-examples/tree/amd-staging/Libraries/libhipcxx>`_

  .. grid-item-card:: API reference

    * :doc:`Standard API <libcudacxx/standard_api>`
    * :doc:`Extended API <libcudacxx/extended_api>`
    * :doc:`Runtime API <libcudacxx/runtime>`
    * :ref:`libhipcxx-limitations`
    * :ref:`libhipcxx-conformance`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
:doc:`Licensing <license>` page.
