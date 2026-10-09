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
  :description: Overview of the libhipcxx mdspan extended API, including memory space accessors, layout_stride_relaxed, restrict_accessor, shared memory accessors, and DLPack conversions in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, mdspan, accessor, restrict_accessor, layout_stride_relaxed, DLPack

.. _libcudacxx-extended-api-mdspan:

Mdspan
======

This page covers the mdspan extended API, providing memory space aware accessors and ``mdspan`` aliases, the ``layout_stride_relaxed`` layout, the *restrict* aliasing policy, and conversions between ``mdspan`` and DLPack.

.. toctree::
   :hidden:
   :maxdepth: 1

   mdspan/host_device_accessor
   mdspan/layout_stride_relaxed
   mdspan/restrict_accessor
   mdspan/shared_memory_accessor
   mdspan/mdspan_to_dlpack
   mdspan/dlpack_to_mdspan

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**

   * - :ref:`host/device/managed mdspan and accessor <libcudacxx-extended-api-mdspan-host-device-accessor>`
     - Host, device, and managed memory space ``mdspan`` and accessors
     - libhipcxx 3.0

   * - :ref:`layout_stride_relaxed <libcudacxx-extended-api-mdspan-layout-stride-relaxed>`
     - Layout mapping policy with negative/zero strides and offset support
     - libhipcxx 3.0

   * - :ref:`restrict mdspan and accessor <libcudacxx-extended-api-mdspan-restrict-accessor>`
     - ``mdspan`` and accessor with the *restrict* aliasing policy
     - libhipcxx 3.0

   * - :ref:`shared_memory mdspan and accessor <libcudacxx-extended-api-mdspan-shared-memory-accessor>`
     - ``mdspan`` and accessor for shared memory
     - libhipcxx 3.4

   * - :ref:`mdspan to dlpack <libcudacxx-extended-api-mdspan-mdspan-to-dlpack>`
     - Convert a ``mdspan`` to a ``DLTensor``
     - libhipcxx 3.4

   * - :ref:`dlpack to mdspan <libcudacxx-extended-api-mdspan-dlpack-to-mdspan>`
     - Convert a ``DLTensor`` to a ``mdspan``
     - libhipcxx 3.4
