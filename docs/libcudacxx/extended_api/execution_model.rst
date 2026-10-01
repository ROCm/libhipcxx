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
  :description: Documentation of the libhipcxx CUDA C++ execution model, covering forward progress guarantees for host threads, device threads, and CUDA API calls.
  :keywords: libhipcxx, ROCm, HIP, C++, execution model, forward progress, device threads, host threads, CUDA API

.. _libcudacxx-extended-api-execution-model:

Execution model
===============

This page documents the CUDA C++ execution model, covering forward progress guarantees for host threads, device threads, and concurrent execution across scopes.

CUDA C++ aims to provide `parallel forward progress [intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__
for all device threads of execution, facilitating the parallelization of pre-existing C++ applications with CUDA C++.

.. dropdown:: `[intro.progress] <https://eel.is/c++draft/intro.progress>`__

    - `[intro.progress.7] <https://eel.is/c++draft/intro.progress#7>`__: For a thread of execution
      providing `concurrent forward progress guarantees <https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees>`__,
      the implementation ensures that the thread will eventually make progress for as long as it has not terminated.

      .. note::

         This applies regardless of whether or not other threads of execution (if any) have been or are making progress.
         To eventually fulfill this requirement means that this will happen in an unspecified but finite amount of time.

    - `[intro.progress.9] <https://eel.is/c++draft/intro.progress>`__: For a thread of execution providing
      `parallel forward progress guarantees <https://eel.is/c++draft/intro.progress#9>`__, the implementation is not required to ensure that
      the thread will eventually make progress if it has not yet executed any execution step; once this thread has executed a step,
      it provides concurrent forward progress guarantees.

      .. note::

         This does not specify a requirement for when to start this thread of execution, which will typically be specified by the entity
         that creates this thread of execution. For example, a thread of execution that provides concurrent forward progress guarantees and executes
         tasks from a set of tasks in an arbitrary order, one after the other, satisfies the requirements of parallel forward progress for these
         tasks.


The CUDA C++ Programming Language is an extension of the C++ Programming Language.
This section documents the modifications and extensions to the `[intro.progress] <https://eel.is/c++draft/intro.progress>`__ section of the current `ISO International Standard ISO/IEC 14882 – Programming Language C++ <https://eel.is/c++draft/>`__ draft.
Modified sections are called out explicitly and their diff is shown in **bold**.
All other sections are additions.

.. _libcudacxx-extended-api-execution-model-host-threads:

Host threads
------------

The forward progress provided by threads of execution created by the host implementation to
execute `main <https://en.cppreference.com/w/cpp/language/main_function>`__, `std::thread <https://en.cppreference.com/w/cpp/thread/thread>`__,
and `std::jthread <https://en.cppreference.com/w/cpp/thread/jthread>`__ is implementation-defined behavior of the host
implementation `[intro.progress] <https://eel.is/c++draft/intro.progress>`__.
General-purpose host implementations should provide concurrent forward progress.

If the host implementation provides `concurrent forward progress [intro.progress.7] <https://eel.is/c++draft/intro.progress#7>`__,
then CUDA C++ provides `parallel forward progress [intro.progress.9] <https://eel.is/c++draft/intro.progress#9>`__ for device threads.


.. _libcudacxx-extended-api-execution-model-device-threads:

Device threads
--------------

Once a device thread makes progress:

- If it is part of a `Cooperative Grid <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73>`__,
  all device threads in its grid shall eventually make progress.
- Otherwise, all device threads in its `thread-block cluster <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters>`__
  shall eventually make progress.

.. note::

   Threads in other thread-block clusters are not guaranteed to eventually make progress.

.. note::

   This implies that all device threads within its thread block shall eventually make progress.


Modify `[intro.progress.1] <https://eel.is/c++draft/intro.progress>`__ as follows (modifications in **bold**):

The implementation may assume that any **host** thread will eventually do one of the following:

- Terminate
- Invoke the function `std::this_thread::yield <https://en.cppreference.com/w/cpp/thread/yield>`__ (`[thread.thread.this] <http://eel.is/c++draft/thread.thread.this>`__)
- Make a call to a library I/O function
- Perform an access through a volatile glvalue
- Perform a synchronization operation or an atomic operation
- Continue execution of a trivial infinite loop (`[stmt.iter.general] <http://eel.is/c++draft/stmt.iter.general>`__)

The implementation may assume that any device thread will eventually do one of the following:

- Terminate
- Make a call to a library I/O function
- Perform an access through a volatile glvalue, except if the designated object has automatic storage duration
- Perform a synchronization operation or an atomic read operation, except if the designated object has automatic storage duration

.. note::

   Some current limitations of device threads relative to host threads
   are implementation defects known to us, that we may fix over time.
   Examples include the undefined behavior that arises from device threads
   that eventually only perform volatile or atomic operations
   on automatic storage duration objects.
   However, other limitations of device threads relative to host threads
   are intentional choices. They enable performance optimizations
   that would not be possible if device threads followed the C++ Standard strictly.
   For example, providing forward progress to programs
   that eventually only perform atomic writes or fences
   would degrade overall performance for little practical benefit.

.. dropdown:: Examples of forward progress guarantee differences between host and device threads due to modifications to `[intro.progress.1] <https://eel.is/c++draft/intro.progress#1>`__.

    The following examples refer to the itemized sub-clauses of the implementation assumptions for host and device threads above
    using "host.threads.<id>" and "device.threads.<id>", respectively.

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Device.0
        // Outcome: grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration.
        __global__ void ex0(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
            if (threadIdx.x == 0) {
                while(atom.load(cuda::memory_order_relaxed) == 0);
            } else if (threadIdx.x == 1) {
                atom.store(1, cuda::memory_order_relaxed);
            }
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Device.1
        // Allowed outcome: No thread makes progress because device threads don't support host.threads.2.
        __global__ void ex1() {
            while(true) cuda::std::this_thread::yield();
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Device.2
        // Allowed outcome: No thread makes progress because device threads don't support host.threads.4
        // for objects with automatic storage duration (see exception in device.threads.3).
        __global__ void ex2() {
            volatile bool True = true;
            while(True);
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Device.3
        // Allowed outcome: No thread makes progress because device threads don't support host.threads.5
        // for objects with automatic storage duration (see exception in device.threads.4).
        __global__ void ex3() {
            cuda::atomic<bool, cuda::thread_scope_thread> True = true;
            while(True.load());
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Device.4
        // Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
        __global void ex4() {
            while(true) { /* empty */ }
        }

.. _libcudacxx-extended-api-execution-model-cuda-apis:

CUDA APIs
---------

A CUDA API call shall eventually either return or ensure at least one device thread makes progress.

CUDA query functions (e.g. `cudaStreamQuery <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435>`__,
`cudaEventQuery <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7>`__, etc.) shall not consistently
return ``cudaErrorNotReady`` without a device thread making progress.

.. note::

   The device thread need not be "related" to the API call, e.g., an API operating on one stream or process may ensure progress of a device thread on another stream or process.

.. note::

   A simple but not sufficient method to test a program for CUDA API Forward Progress conformance is to run them with following environment variables set: ``CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=1``, and then check that the program still terminates.
   If it does not, the program has a bug.
   This method is not sufficient because it does not catch all Forward Progress bugs, but it does catch many such bugs.

.. dropdown:: Examples of CUDA API forward progress guarantees.

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.API.1
        // Outcome: if no other device threads (e.g., from other processes) are making progress,
        // this program terminates and returns cudaSuccess.
        // Rationale: CUDA guarantees that if the device is empty:
        // - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
        // - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
        // - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
        // - Therefore all device threads eventually exit the `hello_world`` grid.
        // - And `cudaDeviceSynchronize`` eventually unblocks.
        __global__ void hello_world() { __syncthreads(); }
        int main() {
            hello_world<<<1,2>>>();
            return (int)cudaDeviceSynchronize();
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.API.2
        // Allowed outcome: eventually, no thread makes progress.
        // Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
        // However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
        // Therefore, the host thread may never be unblocked from the flag spin-loop.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void producer() { flag.store(1); }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            producer<<<1,1>>>();
            while (flag.load() == 0);
            return cudaDeviceSynchronize();
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.API.3
        // Allowed outcome: eventually, no thread makes progress.
        // Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
        // the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void producer() { flag.store(1); }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            producer<<<1,1>>>();
            (void)cudaStreamQuery(0);
            while (flag.load() == 0);
            return cudaDeviceSynchronize();
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.API.4
        // Outcome: terminates.
        // Rationale: same as Execution.Model.API.3, but this example repeatedly calls
        // a CUDA query API in within the flag spin-loop, which guarantees that the device thread
        // eventually makes progress.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void producer() { flag.store(1); }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            producer<<<1,1>>>();
            while (flag.load() == 0) {
                (void)cudaStreamQuery(0);
            }
            return cudaDeviceSynchronize();
        }

.. _libcudacxx-extended-api-execution-model-cuda-dependencies:

Dependencies
~~~~~~~~~~~~

A device thread shall not start until all its dependencies have completed.

.. note::

   Dependencies that prevent device threads from starting to make progress can be created, for example, via `CUDA Stream Commands <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams>`__.
   These may include dependencies on the completion of, among others, `CUDA Events <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events>`__ and `CUDA Kernels <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels>`__.

.. dropdown:: Examples of CUDA API forward progress guarantees due to dependencies

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Stream.0
        // Allowed outcome: eventually, no thread makes progress.
        // Rationale: while CUDA guarantees that one device thread makes progress, since there
        // is no dependency between `first` and `second`, it does not guarantee which thread,
        // and therefore it could always pick the device thread from `second`, which then never
        // unblocks from the spin-loop.
        // That is, `second` may starve `first`.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
        __global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            cudaStream_t s0, s1;
            cudaStreamCreate(&s0);
            cudaStreamCreate(&s1);
            first<<<1,1,0,s0>>>();
            second<<<1,1,0,s1>>>();
            return cudaDeviceSynchronize();
        }

    .. code:: cuda
       :number-lines:

        // Example: Execution.Model.Stream.1
        // Outcome: terminates.
        // Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
        // between first and second, which requires CUDA to run the grids in order.
        cuda::atomic<int, cuda::thread_scope_system> flag = 0;
        __global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
        __global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
        int main() {
            cudaHostRegister(&flag, sizeof(flag));
            cudaStream_t s0;
            cudaStreamCreate(&s0);
            first<<<1,1,0,s0>>>();
            second<<<1,1,0,s0>>>();
            return cudaDeviceSynchronize();
        }
