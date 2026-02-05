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
  :description: Overview of the libhipcxx functional extended API, including maximum, minimum, proclaim_return_type, proclaim_copyable_arguments, and get_device_address utilities.
  :keywords: libhipcxx, ROCm, HIP, C++, functional, maximum, minimum, proclaim_return_type, get_device_address


.. _libcudacxx-extended-api-functional:

Functional
----------

This page covers the functional extended API, including utilities for maximum and minimum computation, return-type proclamation, and obtaining device addresses.

.. toctree::
   :hidden:
   :maxdepth: 1

   functional/proclaim_return_type
   functional/maximum_minimum
   functional/operator_properties
   memory/get_device_address

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::maximum <libcudacxx-extended-api-functional-maximum-minimum>`
     - Returns the maximum of two values
     - libhipcxx 3.0

   * - :ref:`cuda::minimum <libcudacxx-extended-api-functional-maximum-minimum>`
     - Returns the minimum of two values
     - libhipcxx 3.0

   * - :ref:`cuda::proclaim_return_type <libcudacxx-extended-api-functional-proclaim-return-type>`
     - Creates a forwarding call wrapper that proclaims return type
     - libhipcxx 2.7

   * - ``cuda::proclaim_copyable_arguments``
     - Creates a forwarding call wrapper that proclaims that arguments can be freely copied before an invocation of the wrapped callable
     - libhipcxx 3.0

   * - :ref:`cuda::get_device_address <libcudacxx-extended-api-memory-get-device-address>`
     - Returns a valid address to a device object
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::is_associative_v <libcudacxx-extended-api-functional-operator-properties>`, :ref:`cuda::is_commutative_v <libcudacxx-extended-api-functional-operator-properties>`, :ref:`cuda::identity_element() <libcudacxx-extended-api-functional-operator-properties>`, :ref:`cuda::absorbing_element() <libcudacxx-extended-api-functional-operator-properties>`
     - Determines if an operator is associative for a type
     - CCCL 3.3.0
     - CUDA 13.3
