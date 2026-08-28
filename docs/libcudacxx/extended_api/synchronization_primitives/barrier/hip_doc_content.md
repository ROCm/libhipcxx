<!---
    MIT License

    Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
-->

# HIP Barrier Docs

As with the rest of libhipcxx, the main goal of `hip::barrier` is to maintain parity with libcudacxx (now part of the [CCCL umbrella](https://github.com/NVIDIA/cccl)).
The purpose of this document is to highlight where HIP differs from CUDA and to explain common development and porting pitfalls to watch out for.

## Overview
`hip::barrier` is a synchronization primitive used to coordinate selected groups of threads and synchronize memory operations across specified scopes. It implements split barrier semantics, meaning a thread can arrive and continue performing unrelated work up to it's wait call that guards some sensitive resource, unlike `__syncthreads` which block immediately. `hip::barrier` makes two important guarantees when barrier::wait() returns:
1. All threads participating in the barrier have completed their barrier::arrive().
2. All writes by participating threads sequenced before barrier::arrive() at the specified barrier scope (and smaller [see LLVM AMDGPU Backend Sync Scopes](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-llvm-sync-scopes-table)) are visible to all threads participating in the barrier.

### Initialization
`hip::barrier` is default-constructed and must be initialized via the `init()` function before any other use. The barrier must be visible to all participants before it is used, which requires a synchronization after initialization. At block scope, a simple `__syncthreads()` suffices, but for consistency across multiple scopes we recommend using [cooperative_groups'](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/cooperative_groups_reference.html#cooperative-groups) `cooperative_groups::thread_group::sync()` with the appropriate `cooperative_groups::thread_group` subtype where possible. This provides safety guarantees that are hazardous to implement by hand at larger scopes, but does require launching your kernel via the proper [cooperative kernel launch function](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/cooperative_groups_reference.html#cooperative-kernel-launches).

```cpp
__device__ void exampleBarrierInit()
{
  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  auto const threadBlock = cooperative_groups::this_thread_block();

  if (threadBlock.thread_rank() == 0)
  {
    init(&barrier, threadBlock.size());
  }
  threadBlock.sync();
}
```
### Barrier Life Cycle
A barrier has three important internal properties: arrival count, expected arrival count, and phase.
A barrier is initialized with an expected arrival count. On each arrival, the arrival count is updated. When the arrival count reaches the expected count, it resets to the expected arrival count and the phase toggles.
Phase is the property on which the barrier waits — a boolean that flips only when the barrier completes a cycle. `barrier::arrive()` returns the phase captured before the arrival, so the returned token can be passed immediately to `barrier::wait()` to block until that phase completes.

```cpp
__device__ void exampleBarrierLifeCycle()
{
  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  auto const threadBlock = cooperative_groups::this_thread_block();

  if (threadBlock.thread_rank() == 0)
  {
    init(&barrier, threadBlock.size());
  }
  threadBlock.sync();

  // token = 0 because barrier is in its first phase (phase 0)
  hip::barrier<hip::thread_scope_block>::arrival_token token = barrier.arrive();
  // returns when phase != token (i.e. when phase = 1)
  barrier.wait(std::move(token));

  // NOTE: The token local variable is reused, but it is reset by arrive() before being waited on again
  // token = 1 because barrier is in its second phase (phase 1)
  token = barrier.arrive();
  // returns when phase != token (i.e. when phase = 0)
  barrier.wait(std::move(token));

  // token = 0 because barrier is in its third phase (phase 0 again)
  token = barrier.arrive();
  // returns when phase != token (i.e. when phase = 1)
  barrier.wait(std::move(token));
}
```

It is important to carefully manage barrier arrivals and waits to avoid common pitfalls: stale tokens and over-arriving. Notice that in the examples `hip::barrier::wait` takes its argument as an rvalue (i.e. it consumes its argument). While functionally, this is unnecessary as this is a read-only value within the `wait` method, this is a careful API design choice with the intent to express that a phase token is only meant to be waited on once. 

Over-arriving is a bit more nuanced. Over-arriving in odd multiples of the expected count would be equivalent to not flipping the phase at all, after which a subsequent `barrier::wait()` call could deadlock. On the other hand, over-arriving in non-multiples of the expected count is undefined behavior. Put more succintly, a call to `hip::barrier::wait` should only use an arrival token from the current or immediately preceding phase.

#### Wave Divergence

**AMD hardware does not support divergent waves in barrier participation. All such usages will result in undefined behavior (often a deadlock).**
See [Participation Granularity](#participation-granularity).

### Early Exit
In some cases it is useful for threads to exit early while other threads continue with the remaining work. In these cases, the exiting thread must call `barrier::arrive_and_drop()` before returning. This counts as the thread's final arrival and permanently decrements the expected arrival count, so subsequent phases require one fewer arrival to complete.

```cpp
__device__ void exampleBlockReduceKernel(int* output)
{
  constexpr int WAVE_SIZE = 32;
  __shared__ int lds[256]; // Assume initialized to something interesting
  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    init(&barrier, block.size());
  }
  block.sync();

  int tid = static_cast<int>(block.thread_rank());
  // NOTE: cannot split arrive and wait within a wave
  for (int half = static_cast<int>(block.size()) / 2; half >= WAVE_SIZE; half /= 2)
  {
    if (tid < half)
    {
      lds[tid] += lds[tid + half];
      barrier.arrive_and_wait();
    }
    else
    {
      // Thread has no more work; drop from subsequent phases
      barrier.arrive_and_drop();
      return;
    }
  }

  // No need for sync because all threads in a wave are in lock-step
  for (int half = WAVE_SIZE / 2; half > 0; half /= 2)
  {
    if (tid < half)
    {
      lds[tid] += lds[tid + half];
    }
  }

  if (block.thread_rank() == 0)
  {
    output[0] = lds[0];
  }
}
```
### Completion Function
The API supports an optional completion function: a user-provided callable that runs when the barrier completes — after the arrival count update, but before the phase flips and waiting threads are released. Completion functions are supported at all scopes but cannot be combined with asynchronous transaction counting.

```cpp
__device__ void exampleCompletionFunction()
{
  __shared__ int completionCounter;
  auto completionFunction = [&]() {
    completionCounter++;
  };

  using barrier_t = cuda::barrier<cuda::thread_scope_block, decltype(completionFunction)>;
  __shared__ alignas(barrier_t) uint8_t barrierStorage[sizeof(barrier_t)];
  barrier_t* const pBarrier = reinterpret_cast<barrier_t*>(barrierStorage);

  auto const threadBlock = cooperative_groups::this_thread_block();

  if (threadBlock.thread_rank() == 0)
  {
    init(pBarrier, threadBlock.size(), completionFunction);
    completionCounter = 0;
  }
  threadBlock.sync();

  pBarrier->arrive_and_wait();
  // via completionFunc: completionCounter = 1
}
```

> **Note:** The completion function guarantees that memory operations at the specified barrier scope (and smaller scopes) are visible before `barrier::wait()` returns, but it places no restrictions on what operations may be performed inside the completion function itself.

** The following test is an example of an antipattern. This should be avoided in your code.**
```cpp
__device__ bool flag = false;

__device__ void exampleCompletionFunctionHazard()
{
  constexpr int WAVE_SIZE = 32;
  auto const threadBlock = cooperative_groups::this_thread_block();
  auto const threadWave = cooperative_groups::tiled_partition<WAVE_SIZE>(threadBlock);

  auto completionFunction = [&]() {
    // Only the first wave writes the flag thus only the first wave is guaranteed to see the write.
    // All others require a sync to guarantee visibility.
    if (threadWave.meta_group_rank() == 0)
    {
      flag = true;
    }
  };

  using barrier_t = hip::barrier<hip::thread_scope_block, decltype(completionFunction)>;
  __shared__ alignas(barrier_t) uint8_t barrierStorage[sizeof(barrier_t)];
  barrier_t* const pBarrier = reinterpret_cast<barrier_t*>(barrierStorage);

  if (threadBlock.thread_rank() == 0)
  {
    init(pBarrier, threadBlock.size(), completionFunction);
  }
  threadBlock.sync();

  pBarrier->arrive_and_wait();

  if (flag) {
    // Wave 0 is guaranteed to take this path 
  } else {
    // Wave 1 *might* still take this path
  }
}
```

> **Note:** There is no valid, safe case for a system-scope barrier with a completion function. The validity of the completion function is determined by the caller of `init`. If `init` is called on the device, the completion function must be a `__device__` function. If `init` is called on the host, the completion function must be a `__host__` function. (Even a `__host__ __device__` function will have only a single version of the function chose by the compiler.) This means, the address of the completion function is only valid on the hardware that called `init`. If the completion function were to be called on the wrong hardware, the results would be undefined. All of our examples have relied on initializing the barrier on the host in order to make use of the synchronization implications. This means:
- In a heterogenous system (host + device) you can only use a completion function if you have some means of guaranteeing that the host will be the last thread to arrive (thus executing the `__host__` completion function safely).
- In a multi-gpu system (with no host participation in the barrier), we can guarantee that the final arrival will be a device thread, but you would first need to solve the bootstrapping problem via some software defined sync to initialize the barrier on one thread and ensure all participants have visibility before continuing.

### Tracking Asynchronous Memory Operations
At block scope, an eligible ordinary `cuda::memcpy_async(..., bar)` operation
can contribute copy completion to the barrier phase. `wait`, `wait_parity`, and
the `try_wait*` family do not report that phase complete until the participating
arrivals and eligible copy work are complete.

The [HIP Barrier-Backed `memcpy_async`](../__memcpy_async/hip_memcpy_async_doc_content.md)
guide defines eligibility, acceleration and fallback behavior, copy-completion
semantics, participation requirements, and CUDA APIs unavailable on HIP.

## Differences Between HIP and CUDA

### Participation Granularity
AMD hardware does not support barrier participation at a granularity smaller than the size of a wavefront. Threads in a wavefront operate completely in lock-step, so improper usage can result in deadlock as the hardware waits on threads that will never arrive (as they too are waiting).

** The following test is an example of an antipattern. This should be avoided in your code.**
```cpp
__device__ void exampleParticipationGranularityHazard()
{
  __shared__ hip::barrier<hip::thread_scope_block> barrier;
  if (threadIdx.x == 0)
  {
    init(&barrier, blockDim.x);
  }
  __syncthreads();

  constexpr int WAVE_SIZE    = 32;
  constexpr int SUBWAVE_SIZE = WAVE_SIZE - 1;

  // ERR: Threads in the same wavefront take different branches,
  //      so neither branch can gather enough arrivals — deadlock.
  if (threadIdx.x < SUBWAVE_SIZE)
  {
    barrier.arrive_and_wait();
  }
  else
  {
    barrier.arrive_and_wait();
  }
}
```
