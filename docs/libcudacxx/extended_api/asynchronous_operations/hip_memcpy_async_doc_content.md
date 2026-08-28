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

# HIP Barrier-Backed `memcpy_async`

This guide is the asynchronous-copy companion to
[`hip_doc_content.md`](../__barrier/hip_doc_content.md). It documents the narrow
HIP surface supported by `cuda::memcpy_async(..., cuda::barrier&)`; it is not a
general CUDA asynchronous-copy or TMA porting guide.

`cuda::memcpy_async` expresses data movement coordinated with barrier
completion. Issue a copy, perform only work that does not read its destination,
then wait before consuming the copied data. HIP supports this completion-coupled
pattern for a narrow set of ordinary `cuda::memcpy_async(..., bar)` calls; other
valid calls may be drained or synchronous.

## Completion Contract

For a supported ordinary barrier-backed copy, HIP privately registers one
completion event for each participating caller's queued copy work. A barrier
phase completes only after:

1. all expected participant arrivals have occurred; and
2. all private copy-completion events bound to that phase have completed.

Applications must not use CUDA transaction-count APIs to manage ordinary HIP 
copies.

## Usage Patterns

### Single Participant

Use a shared, block-scope barrier to wait for a supported copy from global memory
to LDS. This minimal example has one barrier participant, so it initializes the
barrier with one expected arrival and does not need a post-initialization block
synchronization.

```cpp
#include <cuda/barrier>

constexpr int k_count = 4;

__device__ __attribute__((aligned(16))) int g_src[k_count] = {21, 22, 23, 24};

__device__ int copy_and_consume()
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ __attribute__((aligned(16))) int dest[k_count];

  init(&bar, 1);

  cuda::memcpy_async(
    dest, g_src, cuda::aligned_size_t<16>(sizeof(dest)), bar);

  bar.arrive_and_wait();

  for (int index = 0; index < k_count; ++index) {
    if (dest[index] != g_src[index]) {
      return 1;
    }
  }
  return 0;
}
```

The block-wide examples use the completion contract defined above. Initialize
the barrier once, synchronize so every participant sees the initialized state,
issue the copies, then have every barrier participant arrive and wait before
reading the destination.

### Block-Wide Per-Thread Copies

Every active caller of the non-group overload issues an independent copy. Each
caller therefore needs a distinct non-overlapping source and destination range.

```cpp
#include <cuda/barrier>

__device__ int copy_per_thread(int const* src)
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int dest[];

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  cuda::memcpy_async(&dest[threadIdx.x], &src[threadIdx.x], sizeof(int), bar);

  bar.arrive_and_wait();
  return dest[threadIdx.x] == src[threadIdx.x] ? 0 : 1;
}
```

### Cooperative Block Copy

The cooperative overload partitions one logical copy across the supplied group.
Every thread represented by the group must execute the overload with matching
arguments. On AMD, the group call and barrier participation must be
wave-convergent: no subset of a wave may take a different path around the copy
or barrier.

```cpp
#include <cuda/barrier>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ int copy_with_block(cg::thread_block block, int const* src)
{
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  extern __shared__ int dest[];

  if (block.thread_rank() == 0) {
    init(&bar, block.size());
  }
  block.sync();

  cuda::memcpy_async(block, dest, src, sizeof(int) * block.size(), bar);

  bar.arrive_and_wait();
  return dest[block.thread_rank()] == src[block.thread_rank()] ? 0 : 1;
}
```

## Support and Fallbacks

The ordinary API is supported independently from acceleration. A valid call has
one of the following outcomes:

| Outcome | Exact conditions | Copy and barrier behavior |
| --- | --- | --- |
| Phase-coupled acceleration | On gfx1250, the byte count is nonzero, one endpoint is LDS and the other is global memory, and `bar` is a default-completion block-scope barrier located in LDS. | HIP issues the hardware-assisted global-to-LDS or LDS-to-global copy and privately binds one completion event per caller to the barrier phase. `wait()` does not complete until both the normal arrivals and copy events complete. |
| Drained acceleration | The copy is a nonzero global-to-LDS or LDS-to-global transfer on gfx1250, but `bar` is not the LDS-resident default-completion block barrier above. | HIP issues the same hardware-assisted copy, then drains its async group before `memcpy_async` returns. The barrier has no private copy event, so there is no copy work left to overlap with work between `memcpy_async` and `wait()`. |
| Synchronous fallback | The byte count is zero, both endpoints are LDS, neither endpoint is LDS, or the target lacks the LDS phase-object support used on gfx1250. | HIP performs the ordinary synchronous fallback. The barrier has no private copy event and waiting supplies only normal barrier synchronization. |
| Unavailable on HIP | The operation uses a CUDA byte-counted transaction API, a barrier-coupled bulk-copy API, or a TMA API. | The API is not exposed on HIP. |

The copy object types must be [trivially copyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable). Source and destination ranges must be valid,
non-null, and non-overlapping, consistent with
[`cuda::memcpy_async`'s C++ API contract](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html).

`cuda::aligned_size_t<N>` retains its C++ shape contract on HIP: both endpoints
must be aligned to `N`, and the byte count must be a multiple of `N`; otherwise
the behavior is undefined. HIP does not dynamically validate that promise, and
its accelerated copy helper can copy arbitrary byte counts. The alignment shape
is therefore a caller contract and optimization hint, not a runtime check. Use
the ordinary size overload when the `N`-alignment and size-multiple promise
cannot be made.

## Participation Pitfalls

For a non-group overload, each active caller issues its own copy. Their ranges
must not overlap.

For a cooperative overload, all threads represented by the group must call the
overload with matching arguments. On AMD, keep barrier and group participation
wave-convergent; divergent wave participation is undefined behavior and can
deadlock.

If data copied by one thread will be consumed by other threads, every intended
consumer must participate in the barrier phase before reading the destination.

## CUDA Differences

The following CUDA byte-counted or PTX-coupled surfaces are unavailable on HIP:

- `cuda::device::barrier_expect_tx`;
- `cuda::device::barrier_arrive_tx`;
- `cuda::device::memcpy_async_tx`; and
- barrier-coupled CUDA bulk-copy and TMA APIs.

Use ordinary `cuda::memcpy_async(..., bar)` with normal barrier arrival and wait
for the supported HIP surface. The broader HIP porting matrix and migration
guidance are maintained separately in `docs/barrier_memcpy_async_porting_guide.md`.
