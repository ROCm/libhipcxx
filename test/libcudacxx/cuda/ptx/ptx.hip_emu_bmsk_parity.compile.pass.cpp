//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// HIP-only parity test: verify the HIP software emulation of PTX
// `bmsk.{clamp,wrap}.b32` against hand-computed reference values
// derived from the PTX manual (PTX ISA 7.6, SM_70+):
//
//   bmsk_clamp(a, b): produces a 32-bit mask of `min(b, 32 - (a & 31))`
//   consecutive set bits starting at bit position `a & 31`. If the
//   requested run would extend past bit 31, the run is clamped.
//
//   bmsk_wrap(a, b):  produces a 32-bit mask of `b` consecutive set
//   bits where the run wraps around bit 31 -> bit 0 (i.e. a left
//   rotation by `a & 31` of the low-`b` mask). `b == 0` produces 0;
//   `b >= 32` produces all-ones (0xFFFFFFFF).
//
// All checks are `static_assert`s on constant-evaluated calls into
// `::cuda::ptx::bmsk_{clamp,wrap}` so this is a compile-only test --
// no runtime, no device. The wrappers are marked `_CCCL_API constexpr`
// in <cuda/__ptx/instructions/bmsk.h> on the HIP arm specifically
// to make this verification possible.
//
// Build/run as `compile.pass.cpp` under the HIP backend only:
// upstream NV's bmsk_clamp/bmsk_wrap are device-only inline asm and
// not constexpr-callable from host, so the same parity assertions
// would fail to compile there.

// REQUIRES: hipcc
// UNSUPPORTED: nvcc, nvrtc

// NOTE(HIP/AMD): the public umbrella <cuda/ptx> hard-errors on AMD
// (cuda::ptx is an NV-only public API per its file-header NOTE), so
// route through the individual instruction header which carries the
// HIP software-emulated implementation behind a _CCCL_HIP_COMPILATION()
// gate. Before commit 8657ee7093 the lit harness didn't propagate
// __HIP_PLATFORM_AMD__=1, so <cuda/ptx> silently compiled through and
// _CCCL_HIP_COMPILATION() evaluated to 0 (rendering this entire parity
// test a no-op pass: all the static_asserts below were skipped); the
// per-op header is required now that both gates are wired correctly.
#include <cuda/__ptx/instructions/bmsk.h>
#include <cuda/std/cstdint>

#if _CCCL_HIP_COMPILATION()

// ---- bmsk_clamp ----
// Trivial: b == 0 produces 0 regardless of a.
static_assert(::cuda::ptx::bmsk_clamp(0u, 0u) == 0u);
static_assert(::cuda::ptx::bmsk_clamp(17u, 0u) == 0u);
static_assert(::cuda::ptx::bmsk_clamp(31u, 0u) == 0u);

// b == 32 produces all-ones when a == 0; otherwise clamped.
static_assert(::cuda::ptx::bmsk_clamp(0u, 32u) == 0xFFFFFFFFu);
static_assert(::cuda::ptx::bmsk_clamp(1u, 32u) == 0xFFFFFFFEu); // 31 bits at position 1
static_assert(::cuda::ptx::bmsk_clamp(16u, 32u) == 0xFFFF0000u); // 16 bits at position 16

// Generic non-clamping cases.
static_assert(::cuda::ptx::bmsk_clamp(0u, 1u) == 0x00000001u);
static_assert(::cuda::ptx::bmsk_clamp(0u, 4u) == 0x0000000Fu);
static_assert(::cuda::ptx::bmsk_clamp(4u, 4u) == 0x000000F0u);
static_assert(::cuda::ptx::bmsk_clamp(8u, 8u) == 0x0000FF00u);
static_assert(::cuda::ptx::bmsk_clamp(16u, 16u) == 0xFFFF0000u);

// Clamping cases: start + b > 32 -> truncated at bit 31.
static_assert(::cuda::ptx::bmsk_clamp(28u, 8u) == 0xF0000000u); // 4 bits at 28..31
static_assert(::cuda::ptx::bmsk_clamp(30u, 16u) == 0xC0000000u); // 2 bits at 30..31
static_assert(::cuda::ptx::bmsk_clamp(31u, 16u) == 0x80000000u); // 1 bit at 31

// 'a' is taken mod 32.
static_assert(::cuda::ptx::bmsk_clamp(32u, 4u) == ::cuda::ptx::bmsk_clamp(0u, 4u));
static_assert(::cuda::ptx::bmsk_clamp(33u, 4u) == ::cuda::ptx::bmsk_clamp(1u, 4u));
static_assert(::cuda::ptx::bmsk_clamp(63u, 1u) == ::cuda::ptx::bmsk_clamp(31u, 1u));

// b values beyond 32 are clamped to 32 (no benefit from larger).
static_assert(::cuda::ptx::bmsk_clamp(0u, 33u) == 0xFFFFFFFFu);
static_assert(::cuda::ptx::bmsk_clamp(0u, 64u) == 0xFFFFFFFFu);

// ---- bmsk_wrap ----
// b == 0 always produces 0.
static_assert(::cuda::ptx::bmsk_wrap(0u, 0u) == 0u);
static_assert(::cuda::ptx::bmsk_wrap(17u, 0u) == 0u);

// b == 32 always produces all-ones (entire word) regardless of a.
static_assert(::cuda::ptx::bmsk_wrap(0u, 32u) == 0xFFFFFFFFu);
static_assert(::cuda::ptx::bmsk_wrap(8u, 32u) == 0xFFFFFFFFu);
static_assert(::cuda::ptx::bmsk_wrap(31u, 32u) == 0xFFFFFFFFu);

// b >= 32 also produces all-ones (over-large b).
static_assert(::cuda::ptx::bmsk_wrap(0u, 33u) == 0xFFFFFFFFu);
static_assert(::cuda::ptx::bmsk_wrap(15u, 64u) == 0xFFFFFFFFu);

// Same starting cases as clamp when no wrap-around occurs (low-bit run).
static_assert(::cuda::ptx::bmsk_wrap(0u, 1u) == 0x00000001u);
static_assert(::cuda::ptx::bmsk_wrap(0u, 4u) == 0x0000000Fu);
static_assert(::cuda::ptx::bmsk_wrap(0u, 8u) == 0x000000FFu);

// Non-wrapping runs at non-zero start (same as clamp).
static_assert(::cuda::ptx::bmsk_wrap(4u, 4u) == 0x000000F0u);
static_assert(::cuda::ptx::bmsk_wrap(8u, 8u) == 0x0000FF00u);
static_assert(::cuda::ptx::bmsk_wrap(16u, 16u) == 0xFFFF0000u);

// Wrap-around cases: b extends past bit 31, wraps to bit 0.
//   bmsk_wrap(28, 8): 8 consecutive set bits starting at 28 ->
//     bits 28..31 set (high nibble) + bits 0..3 set (low nibble).
static_assert(::cuda::ptx::bmsk_wrap(28u, 8u) == 0xF000000Fu);
//   bmsk_wrap(30, 4): bits 30..31 + bits 0..1 -> 0xC0000003.
static_assert(::cuda::ptx::bmsk_wrap(30u, 4u) == 0xC0000003u);
//   bmsk_wrap(31, 2): bit 31 + bit 0 -> 0x80000001.
static_assert(::cuda::ptx::bmsk_wrap(31u, 2u) == 0x80000001u);
//   bmsk_wrap(16, 24): bits 16..31 (16 high bits) + bits 0..7 (8 low
//     bits) -> 0xFFFF00FF.
static_assert(::cuda::ptx::bmsk_wrap(16u, 24u) == 0xFFFF00FFu);

// 'a' is taken mod 32 in wrap variant too.
static_assert(::cuda::ptx::bmsk_wrap(32u, 4u) == ::cuda::ptx::bmsk_wrap(0u, 4u));
static_assert(::cuda::ptx::bmsk_wrap(33u, 4u) == ::cuda::ptx::bmsk_wrap(1u, 4u));
static_assert(::cuda::ptx::bmsk_wrap(60u, 8u) == ::cuda::ptx::bmsk_wrap(28u, 8u));

// ---- equivalence cross-checks: when no clamp/wrap is needed,
// clamp and wrap produce the same answer.
static_assert(::cuda::ptx::bmsk_clamp(0u, 1u) == ::cuda::ptx::bmsk_wrap(0u, 1u));
static_assert(::cuda::ptx::bmsk_clamp(0u, 16u) == ::cuda::ptx::bmsk_wrap(0u, 16u));
static_assert(::cuda::ptx::bmsk_clamp(8u, 8u) == ::cuda::ptx::bmsk_wrap(8u, 8u));
static_assert(::cuda::ptx::bmsk_clamp(16u, 16u) == ::cuda::ptx::bmsk_wrap(16u, 16u));

#endif // _CCCL_HIP_COMPILATION()

int main(int, char**)
{
  return 0;
}
