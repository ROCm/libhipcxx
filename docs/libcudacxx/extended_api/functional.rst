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

.. _libcudacxx-extended-api-functional:

Functional
----------

.. toctree::
   :hidden:
   :maxdepth: 1

   functional/proclaim_return_type
   functional/get_device_address
   functional/maximum_minimum

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 0

   * - :ref:`cuda::maximum <libcudacxx-extended-api-functional-maximum-minimum>`
     - Returns the maximum of two values
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::minimum <libcudacxx-extended-api-functional-maximum-minimum>`
     - Returns the minimum of two values
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::proclaim_return_type <libcudacxx-extended-api-functional-proclaim-return-type>`
     - Creates a forwarding call wrapper that proclaims return type
     - libhipcxx 1.9.0 / CCCL 2.0.0
     - CUDA 11.8

   * - ``cuda::proclaim_copyable_arguments``
     - Creates a forwarding call wrapper that proclaims that arguments can be freely copied before an invocation of the wrapped callable
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::get_device_address <libcudacxx-extended-api-functional-get-device-address>`
     - Returns a valid address to a device object
     - CCCL 2.8.0
     - CUDA 12.9
