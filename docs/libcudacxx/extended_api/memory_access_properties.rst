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
  :description: Overview of the libhipcxx memory access properties extended API, including access_property, apply_access_property, associate_access_property, and discard_memory.
  :keywords: libhipcxx, ROCm, HIP, C++, memory access properties, access_property, cache eviction, L2 cache


.. _libcudacxx-extended-api-memory-access-properties:

Memory access properties
------------------------

This page covers the memory access properties extended API, providing types and functions for annotating memory accesses with cache residence hints such as access_property and discard_memory.

.. toctree::
   :hidden:
   :maxdepth: 1

   memory_access_properties/access_property
   memory_access_properties/apply_access_property
   memory_access_properties/associate_access_property

..
   memory_access_properties/annotated_ptr

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`
     - Represents a memory access property
     - libhipcxx 1.6.0 / CCCL 2.0.0 /
     - CUDA 11.5

   * - :ref:`cuda::annotated_ptr <libcudacxx-extended-api-memory-access-properties-annotated-ptr>`
     - Binds an access property to a pointer
     - libhipcxx 1.6.0 / CCCL 2.0.0
     - CUDA 11.5
   * - :ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`
     - Applies access property to memory
     - libhipcxx 1.6.0 / CCCL 2.0.0
     - CUDA 11.5

   * - :ref:`cuda::associate_access_property <libcudacxx-extended-api-memory-access-properties-associate-access-property>`
     - Associates access property with raw pointer
     - libhipcxx 1.6.0 / CCCL 2.0.0
     - CUDA 11.5
