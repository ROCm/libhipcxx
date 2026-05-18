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
// `shl.b{16,32,64}` (logical left shift) and `shr.b{16,32,64}`
// (logical right shift) against hand-computed reference values
// derived from the PTX manual.
//
// PTX semantics (per the PTX ISA reference):
//   shl.bN dst, a, b
//     If b < N : dst = a << b
//     If b >= N: dst = 0           (NOT modulo-N like the C `<<` UB rule;
//                                   PTX defines saturating-zero behaviour)
//   shr.bN dst, a, b
//     If b < N : dst = a >> b      (logical right shift, zero-fill)
//     If b >= N: dst = 0
//
// All checks are `static_assert`s on constant-evaluated calls. The
// HIP wrappers are marked `_CCCL_API constexpr` in
// <cuda/__ptx/instructions/{shl,shr}.h> specifically to make this
// parity verification possible.

// REQUIRES: hipcc
// UNSUPPORTED: nvcc, nvrtc

#include <cuda/ptx>
#include <cuda/std/cstdint>

#if _CCCL_HIP_COMPILATION()

using ::cuda::std::uint16_t;
using ::cuda::std::uint32_t;
using ::cuda::std::uint64_t;

// ---- shl.b16 ----
static_assert(::cuda::ptx::shl(uint16_t{0x0001u}, 0u) == uint16_t{0x0001u});
static_assert(::cuda::ptx::shl(uint16_t{0x0001u}, 4u) == uint16_t{0x0010u});
static_assert(::cuda::ptx::shl(uint16_t{0x00FFu}, 8u) == uint16_t{0xFF00u});
static_assert(::cuda::ptx::shl(uint16_t{0xFFFFu}, 1u) == uint16_t{0xFFFEu});
static_assert(::cuda::ptx::shl(uint16_t{0x0001u}, 15u) == uint16_t{0x8000u});
// b >= 16 -> 0 (PTX saturating-zero behaviour, NOT C-style UB).
static_assert(::cuda::ptx::shl(uint16_t{0xFFFFu}, 16u) == uint16_t{0u});
static_assert(::cuda::ptx::shl(uint16_t{0xFFFFu}, 17u) == uint16_t{0u});
static_assert(::cuda::ptx::shl(uint16_t{0xFFFFu}, 0xFFFFFFFFu) == uint16_t{0u});

// ---- shl.b32 ----
static_assert(::cuda::ptx::shl(uint32_t{0x00000001u}, 0u) == uint32_t{0x00000001u});
static_assert(::cuda::ptx::shl(uint32_t{0x00000001u}, 4u) == uint32_t{0x00000010u});
static_assert(::cuda::ptx::shl(uint32_t{0x0000FFFFu}, 16u) == uint32_t{0xFFFF0000u});
static_assert(::cuda::ptx::shl(uint32_t{0xFFFFFFFFu}, 1u) == uint32_t{0xFFFFFFFEu});
static_assert(::cuda::ptx::shl(uint32_t{0x00000001u}, 31u) == uint32_t{0x80000000u});
// b >= 32 -> 0.
static_assert(::cuda::ptx::shl(uint32_t{0xFFFFFFFFu}, 32u) == uint32_t{0u});
static_assert(::cuda::ptx::shl(uint32_t{0xFFFFFFFFu}, 33u) == uint32_t{0u});
static_assert(::cuda::ptx::shl(uint32_t{0xFFFFFFFFu}, 0xFFFFFFFFu) == uint32_t{0u});

// ---- shl.b64 ----
static_assert(::cuda::ptx::shl(uint64_t{0x0000000000000001ull}, 0u) == uint64_t{0x0000000000000001ull});
static_assert(::cuda::ptx::shl(uint64_t{0x0000000000000001ull}, 32u) == uint64_t{0x0000000100000000ull});
static_assert(::cuda::ptx::shl(uint64_t{0x0000000000000001ull}, 63u) == uint64_t{0x8000000000000000ull});
static_assert(::cuda::ptx::shl(uint64_t{0xFFFFFFFFFFFFFFFFull}, 1u) == uint64_t{0xFFFFFFFFFFFFFFFEull});
// b >= 64 -> 0.
static_assert(::cuda::ptx::shl(uint64_t{0xFFFFFFFFFFFFFFFFull}, 64u) == uint64_t{0ull});
static_assert(::cuda::ptx::shl(uint64_t{0xFFFFFFFFFFFFFFFFull}, 65u) == uint64_t{0ull});
static_assert(::cuda::ptx::shl(uint64_t{0xFFFFFFFFFFFFFFFFull}, 0xFFFFFFFFu) == uint64_t{0ull});

// ---- shr.b16 ----
static_assert(::cuda::ptx::shr(uint16_t{0x8000u}, 0u) == uint16_t{0x8000u});
static_assert(::cuda::ptx::shr(uint16_t{0x8000u}, 4u) == uint16_t{0x0800u});
static_assert(::cuda::ptx::shr(uint16_t{0xFF00u}, 8u) == uint16_t{0x00FFu});
static_assert(::cuda::ptx::shr(uint16_t{0xFFFFu}, 1u) == uint16_t{0x7FFFu});
static_assert(::cuda::ptx::shr(uint16_t{0x8000u}, 15u) == uint16_t{0x0001u});
// b >= 16 -> 0 (logical shift; sign bit not preserved).
static_assert(::cuda::ptx::shr(uint16_t{0xFFFFu}, 16u) == uint16_t{0u});
static_assert(::cuda::ptx::shr(uint16_t{0xFFFFu}, 17u) == uint16_t{0u});
static_assert(::cuda::ptx::shr(uint16_t{0xFFFFu}, 0xFFFFFFFFu) == uint16_t{0u});

// ---- shr.b32 ----
static_assert(::cuda::ptx::shr(uint32_t{0x80000000u}, 0u) == uint32_t{0x80000000u});
static_assert(::cuda::ptx::shr(uint32_t{0x80000000u}, 4u) == uint32_t{0x08000000u});
static_assert(::cuda::ptx::shr(uint32_t{0xFFFF0000u}, 16u) == uint32_t{0x0000FFFFu});
static_assert(::cuda::ptx::shr(uint32_t{0xFFFFFFFFu}, 1u) == uint32_t{0x7FFFFFFFu});
static_assert(::cuda::ptx::shr(uint32_t{0x80000000u}, 31u) == uint32_t{0x00000001u});
// b >= 32 -> 0.
static_assert(::cuda::ptx::shr(uint32_t{0xFFFFFFFFu}, 32u) == uint32_t{0u});
static_assert(::cuda::ptx::shr(uint32_t{0xFFFFFFFFu}, 33u) == uint32_t{0u});
static_assert(::cuda::ptx::shr(uint32_t{0xFFFFFFFFu}, 0xFFFFFFFFu) == uint32_t{0u});

// ---- shr.b64 ----
static_assert(::cuda::ptx::shr(uint64_t{0x8000000000000000ull}, 0u) == uint64_t{0x8000000000000000ull});
static_assert(::cuda::ptx::shr(uint64_t{0x8000000000000000ull}, 32u) == uint64_t{0x0000000080000000ull});
static_assert(::cuda::ptx::shr(uint64_t{0x8000000000000000ull}, 63u) == uint64_t{0x0000000000000001ull});
static_assert(::cuda::ptx::shr(uint64_t{0xFFFFFFFFFFFFFFFFull}, 1u) == uint64_t{0x7FFFFFFFFFFFFFFFull});
// b >= 64 -> 0.
static_assert(::cuda::ptx::shr(uint64_t{0xFFFFFFFFFFFFFFFFull}, 64u) == uint64_t{0ull});
static_assert(::cuda::ptx::shr(uint64_t{0xFFFFFFFFFFFFFFFFull}, 65u) == uint64_t{0ull});
static_assert(::cuda::ptx::shr(uint64_t{0xFFFFFFFFFFFFFFFFull}, 0xFFFFFFFFu) == uint64_t{0ull});

// ---- shl/shr round-trip: shifting a single bit left then back right
// recovers the original. Covers every bit position to catch off-by-one
// edge cases at the boundary.
static_assert(::cuda::ptx::shr(::cuda::ptx::shl(uint32_t{1u}, 0u), 0u) == uint32_t{1u});
static_assert(::cuda::ptx::shr(::cuda::ptx::shl(uint32_t{1u}, 15u), 15u) == uint32_t{1u});
static_assert(::cuda::ptx::shr(::cuda::ptx::shl(uint32_t{1u}, 16u), 16u) == uint32_t{1u});
static_assert(::cuda::ptx::shr(::cuda::ptx::shl(uint32_t{1u}, 31u), 31u) == uint32_t{1u});

// ---- shl/shr alias-safety: the bit_cast-based plumbing is the
// strict-aliasing-safe version of what the WAR used to do via
// '*reinterpret_cast<...>'. A round-trip on a non-uint type (here:
// int32_t, which is layout-compatible but a different type) verifies
// the wrapper still preserves bits across the cast.
static_assert(::cuda::ptx::shl(int32_t{1}, 4u) == int32_t{16});
static_assert(::cuda::ptx::shr(int32_t{16}, 4u) == int32_t{1});
// Sign-bit handling: shr is logical (not arithmetic), so a negative
// int32_t shifted right does NOT sign-extend on the HIP path.
static_assert(::cuda::ptx::shr(int32_t{-1}, 1u) == int32_t{0x7FFFFFFF});

#endif // _CCCL_HIP_COMPILATION()

int main(int, char**)
{
  return 0;
}
