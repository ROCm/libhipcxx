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

# CUDA Barrier and Barrier-Backed Async-Copy Porting Guide

This guide is for CUDA device code that uses `cuda::barrier` and its
barrier-backed asynchronous-copy helpers and must also build with libhipcxx on
AMD HIP. The barrier class and its scope/completion specializations are the
primary API surface; async-copy APIs appear below as barrier interoperation.

The [HIP barrier API guide](../include/cuda/__barrier/hip_doc_content.md) and
[HIP barrier-backed `memcpy_async` guide](../include/cuda/__memcpy_async/hip_memcpy_async_doc_content.md)
define the detailed API and copy-completion behavior. This document maps CUDA
source shapes to the HIP surface. `Supported` means the public call is available
with the stated ordering contract. `Unsupported` means it is not exposed by the
HIP public headers.

## Quick Rules

- Ordinary `cuda::memcpy_async(..., barrier)` is supported. Its byte count
  controls the data transfer; eligible phase-coupled copies use one private
  completion event per participating caller.
- `cuda::device::barrier_expect_tx`, `cuda::device::barrier_arrive_tx`, and
  `cuda::device::memcpy_async_tx` are unavailable on HIP. Do not substitute a
  byte count or event count into a different public API.
- CUDA barrier-coupled bulk-copy and TMA APIs are unavailable on HIP.
- A public HIP cluster barrier interface is unsupported. A block-scope barrier
  is not a substitute for a cluster barrier. HIP exposes
  `thread_scope_thread`, `thread_scope_block`, `thread_scope_device`, and
  `thread_scope_system`.

## `cuda::barrier` Class Matrix

The public class is `cuda::barrier<Scope, CompletionFunction>`. The following
matrix describes the actual HIP class shapes rather than treating every
template instantiation as one implementation.

| Barrier class shape | AMD HIP status | Implementation and porting consequence |
| --- | --- | --- |
| `cuda::barrier<cuda::thread_scope_block, __empty_completion>` in `__shared__` storage on gfx1250 | Supported, LDS workgroup specialization | Uses the AMD LDS phase object (`__lds_barrier_t`) for ordinary barrier operations and eligible ordinary barrier-backed copies. `max()` is 65535 for this specialization; the practical HIP launch limit remains 1024 threads per block. |
| CUDA `cuda::barrier<cuda::thread_scope_block, __empty_completion>` in cluster-shared storage | Unsupported on HIP | CUDA can use a block-scope barrier object in cluster-shared storage on supported cluster-capable targets. HIP has no cluster-shared barrier path; its LDS specialization is workgroup-local only. |

### Cluster barrier gap

CUDA cluster programming can use cluster-shared synchronization state and
cluster-scoped operations. The HIP-facing barrier API does not provide that
programmer interface. Do not port a CUDA cluster barrier by changing its scope
to `thread_scope_block`; that changes the participating execution domain and
is not semantic parity.

## Endpoint Status

### Barriers

Outside of cluster scope, the public barrier class and ordinary arrival/wait
operations are supported. Completion functions use the generic/software path
and cannot be combined with the accelerated barrier-backed copy path.

### Scope summary

| CUDA scope | HIP public status | HIP barrier class used |
| --- | --- | --- |
| `cuda::thread_scope_thread` | Supported | Empty completion aliases the block-scope HIP class; custom completion uses the generic software class. |
| `cuda::thread_scope_block` | Supported | Empty completion uses the LDS workgroup class only for shared gfx1250 objects; other cases use the generic/software state. Custom completion always uses the generic/software class. |
| `cuda::thread_scope_device` | Supported | Generic/software barrier state. |
| `cuda::thread_scope_system` | Supported | Generic/software barrier state. |
| CUDA cluster-scoped or cluster-shared barrier use | Unsupported | No HIP public cluster barrier interface or `cuda::barrier` specialization. |

### Barrier Interoperation and Transaction Helpers

| CUDA public endpoint | AMD HIP status | Porting guidance |
| --- | --- | --- |
| `cuda::device::barrier_native_handle(barrier<thread_scope_block>&)` | Supported | Returns a HIP `uint64_t*` handle for supported block-scope barriers. The pointed-to layout is implementation-specific and must not be exchanged between CUDA and HIP implementations. |
| `cuda::device::barrier_expect_tx(barrier&, tx_count)` | Unsupported | Not exposed on HIP. Refactor the CUDA transaction-counted operation rather than replacing its count with an ordinary barrier arrival. |
| `cuda::device::barrier_arrive_tx(barrier&, arrive_count, tx_count)` | Unsupported | Not exposed on HIP. Refactor the CUDA transaction-counted operation rather than replacing `tx_count` with a byte or event count. |

### Barrier-Backed `cuda::memcpy_async`

| CUDA public endpoint | AMD HIP status | Porting guidance |
| --- | --- | --- |
| `cuda::memcpy_async(dst, src, size, barrier)` | Supported | Single-caller ordinary copy. Phase-coupled acceleration requires a nonzero global-to-LDS or LDS-to-global copy on gfx1250 with an LDS-resident default-completion block-scope barrier. Other valid calls are drained or synchronous as described in the detailed guide. |
| `cuda::memcpy_async(group, dst, src, size, barrier)` | Supported | Cooperative ordinary copy with the same acceleration and fallback conditions. Every group member must call with matching arguments, and AMD copy and barrier participation must be wave-convergent. |
| `cuda::memcpy_async(dst, src, aligned_size_t<Alignment>, barrier)` | Supported | Same as the ordinary single-caller overload. Both endpoints must meet `Alignment`, and the byte count must be a multiple of `Alignment`. |
| `cuda::memcpy_async(group, dst, src, aligned_size_t<Alignment>, barrier)` | Supported | Same as the ordinary group overload, with the same alignment and size-multiple contract. |
| `cuda::memcpy_async(dst, src, size, pipeline)` | Unsupported | The CUDA API is defined, but a public HIP pipeline surface is not currently reachable or validated. |
| `cuda::memcpy_async(group, dst, src, size, pipeline)` | Unsupported | Same pipeline limitation as the single-caller overload. |
| `cuda::memcpy_async(dst, annotated_ptr<src>, size, sync)` | Unsupported | CUDA annotated-pointer convenience overload. `sync` may be a barrier or pipeline; this overload is not a validated HIP public surface. |
| `cuda::memcpy_async(annotated_ptr<dst>, annotated_ptr<src>, size, sync)` | Unsupported | CUDA annotated-pointer convenience overload. `sync` may be a barrier or pipeline; this overload is not a validated HIP public surface. |
| `cuda::memcpy_async(group, dst, annotated_ptr<src>, size, sync)` | Unsupported | CUDA annotated-pointer group overload. `sync` may be a barrier or pipeline; this overload is not a validated HIP public surface. |
| `cuda::memcpy_async(group, annotated_ptr<dst>, annotated_ptr<src>, size, sync)` | Unsupported | CUDA annotated-pointer group overload. `sync` may be a barrier or pipeline; this overload is not a validated HIP public surface. |

### Explicit Transaction-Counted and Bulk Copy APIs

| CUDA endpoint | Surface category | AMD HIP status |
| --- | --- | --- |
| `cuda::device::barrier_expect_tx(barrier&, tx_count)` | Public C++ API | Unsupported |
| `cuda::device::barrier_arrive_tx(barrier&, arrive_count, tx_count)` | Public C++ API | Unsupported |
| `cuda::device::memcpy_async_tx(dst, src, aligned_size_t<Alignment>, barrier)` | Public C++ API | Unsupported |
| `cuda::device::experimental::cp_async_bulk_global_to_shared(dst, src, size, barrier)` | Public experimental C++ wrapper over PTX | Unsupported |
| `cuda::device::experimental::cp_async_bulk_shared_to_global(dst, src, size)` | Public experimental C++ wrapper over PTX | Unsupported |
| `cuda::device::experimental::cp_async_bulk_tensor_{1..5}d_global_to_shared(...)` | Public experimental C++ wrappers over PTX | Unsupported |
| `cuda::device::experimental::cp_async_bulk_tensor_{1..5}d_shared_to_global(...)` | Public experimental C++ wrappers over PTX | Unsupported |
| `cuda::device::experimental::fence_proxy_async_shared_cta()` | Public experimental C++ wrapper over PTX | Unsupported |
| `cuda::device::experimental::cp_async_bulk_commit_group()` | Public experimental C++ wrapper over PTX | Unsupported |
| `cuda::device::experimental::cp_async_bulk_wait_group_read<N>()` | Public experimental C++ wrapper over PTX | Unsupported |

## Migration Patterns

### Supported: ordinary barrier-backed copy

```cpp
__shared__ cuda::barrier<cuda::thread_scope_block> bar;

if (threadIdx.x == 0) {
  init(&bar, blockDim.x);
}
__syncthreads();

cuda::memcpy_async(block, shared_dst, global_src, byte_count, bar);
bar.arrive_and_wait();
```

The `byte_count` remains the number of bytes copied. Do not add a separate
transaction-counted API for this call; HIP owns ordinary-copy completion
bookkeeping privately.

### CUDA-only today: explicit byte-counted copy

```cpp
cuda::device::memcpy_async_tx(shared_dst, global_src,
                              cuda::aligned_size_t<16>(byte_count), bar);
cuda::device::barrier_arrive_tx(bar, 1, byte_count);
```

This source shape remains unsupported on HIP. There is no token-for-token HIP
rewrite: do not replace its byte count with an ordinary barrier arrival or a
private completion-event count. Refactor around ordinary
`cuda::memcpy_async(..., bar)` only when that API's completion contract is
appropriate for the operation.

## Scope and Caveats

- This matrix covers the public CUDA barrier, barrier-coupled async-copy,
  explicit async-copy, bulk-copy, as exposed by the relevant `<cuda/barrier>` and `<cuda/memcpy_async>` API families.
- Check target architecture, storage placement, direction, and byte count before
  relying on phase-coupled ordinary-copy completion. The eligible HIP shape is a
  nonzero global-to-LDS or LDS-to-global copy on gfx1250 using an LDS-resident
  default-completion block-scope barrier.