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

.. _libcudacxx-extended-api-asynchronous-operations:

Asynchronous Operations
-----------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   asynchronous_operations/memcpy_async_tx
   asynchronous_operations/memcpy_async

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>`
     - Asynchronously copies one range to another
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - :ref:`cuda::memcpy_async_tx <libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx>`
     - Asynchronously copies one range to another with manual transaction accounting
     - libhipcxx 1.2.0 / CCCL 2.0.0 / CUDA 11.1

.. note::

  **Asynchronous operations** like `memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>`
  are non-blocking operations performed as-if by a new thread of execution.
