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

.. _libcudacxx-extended-api-memory-access-properties:

Memory access properties
------------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   memory_access_properties/access_property
   memory_access_properties/annotated_ptr
   memory_access_properties/apply_access_property
   memory_access_properties/associate_access_property
   memory_access_properties/discard_memory

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`
     - Represents a memory access property
     - libhipcxx 1.6.0 / CCCL 2.0.0 / CUDA 11.5
   * - :ref:`cuda::annotated_ptr <libcudacxx-extended-api-memory-access-properties-annotated-ptr>`
     - Binds an access property to a pointer
     - libhipcxx 1.6.0 / CCCL 2.0.0 / CUDA 11.5
   * - :ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`
     - Applies access property to memory
     - libhipcxx 1.6.0 / CCCL 2.0.0 / CUDA 11.5
   * - :ref:`cuda::associate_access_property <libcudacxx-extended-api-memory-access-properties-associate-access-property>`
     - Associates access property with raw pointer
     - libhipcxx 1.6.0 / CCCL 2.0.0 / CUDA 11.5
   * - :ref:`cuda::discard_memory <libcudacxx-extended-api-memory-access-properties-discard-memory>`
     - Writes indeterminate values to memory
     - libhipcxx 1.6.0 / CCCL 2.0.0 / CUDA 11.5
