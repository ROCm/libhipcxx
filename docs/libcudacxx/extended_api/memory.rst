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
  :description: Overview of the libhipcxx memory extended API, including pointer alignment, address space queries, pointer range checks, and device address utilities in libhipcxx for HIP.
  :keywords: libhipcxx, ROCm, HIP, C++, memory, align_up, align_down, is_aligned, ptr_rebind, get_device_address

.. _libcudacxx-extended-api-memory:

Memory
======

This page covers the memory extended API, providing pointer alignment, address space, and pointer range utilities.

``cuda::aligned_size_t`` and ``cuda::discard_memory`` are not supported in libhipcxx. See :ref:`libhipcxx-limitations`.

.. toctree::
   :hidden:
   :maxdepth: 1

   memory/align_down
   memory/align_up
   memory/get_device_address
   memory/is_address_from
   memory/is_aligned
   memory/ptr_rebind
   memory/ptr_in_range
   memory/ranges_overlap
   memory/is_pointer_accessible

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst. The pages are
   also listed in exclude_patterns in docs/conf.py.

   - cuda::aligned_size_t is only used by cuda::memcpy_async and <cuda/pipeline>.
   - cuda::discard_memory is a no-op on AMD GPUs.

   memory/aligned_size
   memory/discard_memory

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **libhipcxx Availability**

   * - :ref:`get_device_address <libcudacxx-extended-api-memory-get-device-address>`
     - Returns a valid address to a device object
     - libhipcxx 3.0 (in ``<cuda/memory>`` since libhipcxx 3.4)

   * - :ref:`is_address_from and is_object_from <libcudacxx-extended-api-memory-is_address_from>`
     - Check if a pointer or object is from a specific address space
     - libhipcxx 3.4

   * - :ref:`is_aligned <libcudacxx-extended-api-memory-is_aligned>`
     - Check if a pointer is aligned
     - libhipcxx 3.4

   * - :ref:`align_up <libcudacxx-extended-api-memory-align_up>`
     - Align up a pointer to the specified alignment
     - libhipcxx 3.4

   * - :ref:`align_down <libcudacxx-extended-api-memory-align_down>`
     - Align down a pointer to the specified alignment
     - libhipcxx 3.4

   * - :ref:`ptr_rebind <libcudacxx-extended-api-memory-ptr_rebind>`
     - Rebind a pointer to a different type
     - libhipcxx 3.4

   * - :ref:`ptr_in_range <libcudacxx-extended-api-memory-ptr_in_range>`
     - Check if a pointer is in a range
     - libhipcxx 3.4

   * - :ref:`ranges_overlap <libcudacxx-extended-api-memory-ranges_overlap>`
     - Check if two ranges overlap
     - libhipcxx 3.4

   * - :ref:`is_host_accessible <libcudacxx-extended-api-memory-is_pointer_accessible>`, :ref:`is_device_accessible <libcudacxx-extended-api-memory-is_pointer_accessible>`, :ref:`is_managed <libcudacxx-extended-api-memory-is_pointer_accessible>`
     - Check if a pointer is accessible from the host, device, or managed memory
     - libhipcxx 3.4

..
   Not supported in libhipcxx, see reference/libhipcxx-limitations.rst.

   * - :ref:`aligned_size_t <libcudacxx-extended-api-memory-aligned-size>`
     - Defines an extent of bytes with a statically defined alignment.
     - libhipcxx 2.7 (in ``<cuda/memory>`` since libhipcxx 3.4)

   * - :ref:`discard_memory <libcudacxx-extended-api-memory-discard-memory>`
     - Writes indeterminate values to memory
     - libhipcxx 2.7 (in ``<cuda/memory>`` since libhipcxx 3.4)
