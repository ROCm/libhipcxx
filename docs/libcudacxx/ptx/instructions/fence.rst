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

.. _libcudacxx-ptx-instructions-fence:

fence
=====

This page documents the ``cuda::ptx`` wrappers for the fence PTX instruction, which enforces memory ordering across specified thread scopes.

-  PTX ISA:
   `fence <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence>`__

.. _fence-1:

fence
-----

.. include:: generated/fence.rst

fence.sync_restrict
-------------------

.. include:: generated/fence_sync_restrict.rst

fence.mbarrier_init
-------------------

.. include:: generated/fence_mbarrier_init.rst

fence.proxy.alias
-----------------

.. include:: generated/fence_proxy_alias.rst

fence.proxy.async
-----------------


.. include:: generated/fence_proxy_async.rst

fence.proxy.async.sync_restrict
-------------------------------

.. include:: generated/fence_proxy_async_generic_sync_restrict.rst

fence.proxy.tensormap
---------------------

.. include:: generated/fence_proxy_tensormap_generic.rst
