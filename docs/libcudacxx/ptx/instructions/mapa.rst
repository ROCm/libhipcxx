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

.. _libcudacxx-ptx-instructions-mapa:

mapa
====

-  PTX ISA:
   `mapa <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa>`__

This instruction can `currently not be
implemented <https://github.com/NVIDIA/cccl/issues/1414>`__ by libhipcxx.
The instruction can be accessed through the cooperative groups
`cluster_group <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group>`__
API:

Usage:
------

.. code:: cuda

   #include <cooperative_groups.h>

   __cluster_dims__(2)
   __global__ void kernel() {
       __shared__ int x;
       x = 1;
       namespace cg = cooperative_groups;
       cg::cluster_group cluster = cg::this_cluster();

       cluster.sync();

       // Get address of remote shared memory value:
       unsigned int other_block_rank = cluster.block_rank() ^ 1;
       int * remote_x = cluster.map_shared_rank(&bar, other_block_rank);

       // Write to remote value:
       *remote_x = 2;
   }
