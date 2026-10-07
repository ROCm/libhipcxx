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

.. _libcudacxx-ptx-instructions-barrier-cluster:

barrier.cluster
===============

This page documents the ``cuda::ptx`` wrappers for the barrier.cluster PTX instruction, which synchronizes all threads across a thread block cluster.

-  PTX ISA:
   `barrier.cluster <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster>`__

Similar functionality is provided through the builtins
``__cluster_barrier_arrive(), __cluster_barrier_arrive_relaxed(), __cluster_barrier_wait()``,
as well as the ``cooperative_groups::cluster_group``
`API <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group>`__.

The ``.aligned`` variants of the instructions are not exposed.

.. include:: generated/barrier_cluster.rst
