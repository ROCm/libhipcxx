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

.. _libcudacxx-extended-api-synchronization:

Synchronization Primitives
===========================

.. toctree::
   :hidden:
   :maxdepth: 1

   synchronization_primitives/atomic
   synchronization_primitives/atomic_ref
   synchronization_primitives/latch
   synchronization_primitives/barrier
   synchronization_primitives/counting_semaphore
   synchronization_primitives/binary_semaphore
   synchronization_primitives/pipeline

.. rubric:: Atomics

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::atomic <libcudacxx-extended-api-synchronization-atomic>`
     - System-wide `std::atomic <https://en.cppreference.com/w/cpp/atomic/atomic>`_ objects and operations
     - libhipcxx 1.0.0 / CCCL 2.0.0 / CUDA 10.2
   * - :ref:`cuda::atomic_ref <libcudacxx-extended-api-synchronization-atomic-ref>`
     - System-wide `std::atomic_ref <https://en.cppreference.com/w/cpp/atomic/atomic_ref>`_ objects and operations
     - libhipcxx 1.7.0 / CCCL 2.0.0 / CUDA 11.6

.. rubric:: Latches

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::latch <libcudacxx-extended-api-synchronization-latch>`
     - System-wide `std::latch <https://en.cppreference.com/w/cpp/thread/latch>`_ single-phase asynchronous
       thread coordination mechanism
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0

.. rubric:: Barriers

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`
     - System wide `std::barrier <https://en.cppreference.com/w/cpp/thread/barrier>`_ multi-phase asynchronous
       thread coordination mechanism
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0

.. rubric:: Semaphores

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::counting_semaphore <libcudacxx-extended-api-synchronization-counting-semaphore>`
     - System wide `std::counting_semaphore <https://en.cppreference.com/w/cpp/thread/counting_semaphore>`_
       primitive for constraining concurrent access
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - :ref:`cuda::binary_semaphore <libcudacxx-extended-api-synchronization-counting-semaphore>`
     - System wide `std::binary_semaphore <https://en.cppreference.com/w/cpp/thread/counting_semaphore>`_
       primitive for mutual exclusion
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0

.. rubric:: Pipelines

The pipeline library is included in the CUDA Toolkit, but is not part of the open source libhipcxx distribution.

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::pipeline <libcudacxx-extended-api-synchronization-pipeline>`
     - Coordination mechanism for sequencing asynchronous operations
     - libhipcxx 1.2.0 / CCCL 2.0.0 / CUDA 11.1
   * - :ref:`cuda::pipeline_shared_state <libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state>`
     - :ref:`cuda::pipeline <libcudacxx-extended-api-synchronization-pipeline>` shared state object
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - :ref:`cuda::pipeline_role <libcudacxx-extended-api-synchronization-pipeline-pipeline-role>`
     - Defines producer/consumer role for a thread participating in a *pipeline*
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - :ref:`cuda::make_pipeline <libcudacxx-extended-api-synchronization-pipeline-pipeline-role>`
     - Creates a :ref:`cuda::pipeline <libcudacxx-extended-api-synchronization-pipeline>`
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - :ref:`cuda::pipeline_consumer_wait_prior <libcudacxx-extended-api-synchronization-pipeline-pipeline-consumer-wait-prior>`
     - Blocks the current thread until all operations committed up to a prior *pipeline stage* complete
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - :ref:`cuda::pipeline_producer_commit <libcudacxx-extended-api-synchronization-pipeline-pipeline-producer-commit>`
     - Binds operations previously issued by the current thread to a :ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`
     - libhipcxx 1.1.0 / CCCL 2.0.0 / CUDA 11.0
