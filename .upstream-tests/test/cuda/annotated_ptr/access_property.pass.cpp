//===----------------------------------------------------------------------===//
//
// Part of libhip++ (derived from libcu++),
// the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: c++98, c++03

#include "utils.h"

__device__ __host__ __noinline__
void test_access_property_interleave() {
    const uint64_t INTERLEAVE_NORMAL           = uint64_t{0x10F0000000000000};
    const uint64_t INTERLEAVE_NORMAL_DEMOTE    = uint64_t{0x16F0000000000000};
    const uint64_t INTERLEAVE_PERSISTING       = uint64_t{0x14F0000000000000};
    const uint64_t INTERLEAVE_STREAMING        = uint64_t{0x12F0000000000000};
    hip::access_property ap(hip::access_property::persisting{});
    hip::access_property ap2;

    assert(INTERLEAVE_PERSISTING == static_cast<uint64_t>(ap));
    assert(static_cast<uint64_t>(ap2) == INTERLEAVE_NORMAL);

    ap = hip::access_property(hip::access_property::normal());
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL_DEMOTE);

    ap = hip::access_property(hip::access_property::streaming());
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_STREAMING);

    ap = hip::access_property(hip::access_property::normal(), 2.0f);
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL_DEMOTE);
}

__device__ __host__ __noinline__
void test_access_property_block() {
    //assuming ptr address is 0;
    const size_t TOTAL_BYTES = 0xFFFFFFFF;
    const size_t HIT_BYTES = 0xFFFFFFFF;
    const size_t BLOCK_0ADDR_PERSISTHIT_STREAMISS_MAXBYTES = size_t{0x1DD00FE000000000};
    const uint64_t INTERLEAVE_NORMAL = uint64_t{0x10F0000000000000};

    hip::access_property ap(0x0, HIT_BYTES, TOTAL_BYTES, hip::access_property::persisting{}, hip::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == BLOCK_0ADDR_PERSISTHIT_STREAMISS_MAXBYTES);

    ap = hip::access_property(0x0, 0xFFFFFFFF, 0xFFFFFFFFF, hip::access_property::persisting{}, hip::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL);

    ap = hip::access_property(0x0, 0xFFFFFFFFF, 0xFFFFFFFF, hip::access_property::persisting{}, hip::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL);

    ap = hip::access_property(0x0, 0, 0, hip::access_property::persisting{}, hip::access_property::streaming{});
    assert(static_cast<uint64_t>(ap) == INTERLEAVE_NORMAL);

    for (size_t ptr = 1; ptr < size_t{0xFFFFFFFF}; ptr <<= 1) {
        for (size_t hit = 1; hit < size_t{0xFFFFFFFF}; hit <<= 1) {
            ap = hip::access_property((void*)ptr, hit, hit, hip::access_property::persisting{}, hip::access_property::streaming{});
            DPRINTF("Block encoding PTR:%p, hit:%p, block encoding:%p\n", ptr, hit, static_cast<uint64_t>(ap));
        }
    }
}

template<typename T>
__host__ __device__ __noinline__ void test_global_implicit_property(T ap, cudaAccessProperty cp) {
  // Test implicit conversions
  cudaAccessProperty v = ap;
  assert(cp == v);

  // Test default, copy constructor, and copy-assignent
  hip::access_property o(ap);
  hip::access_property d;
  d = ap;

  // Test explicit conversion to i64
  std::uint64_t x = (std::uint64_t)o;
  std::uint64_t y = (std::uint64_t)d;
  assert(x == y);
}

__host__ __device__ __noinline__ void test_global() {
  hip::access_property o(hip::access_property::global{});
  std::uint64_t x = (std::uint64_t)o;
  unused(x);
}

__host__ __device__ __noinline__ void test_shared() {
  (void)hip::access_property::shared{};
}

static_assert(sizeof(hip::access_property::shared) == 1, "");
static_assert(sizeof(hip::access_property::global) == 1, "");
static_assert(sizeof(hip::access_property::persisting) == 1, "");
static_assert(sizeof(hip::access_property::normal) == 1, "");
static_assert(sizeof(hip::access_property::streaming) == 1, "");
static_assert(sizeof(hip::access_property) == 8, "");

static_assert(alignof(hip::access_property::shared) == 1, "");
static_assert(alignof(hip::access_property::global) == 1, "");
static_assert(alignof(hip::access_property::persisting) == 1, "");
static_assert(alignof(hip::access_property::normal) == 1, "");
static_assert(alignof(hip::access_property::streaming) == 1, "");
static_assert(alignof(hip::access_property) == 8, "");

int main(int argc, char ** argv)
{
  test_access_property_interleave();
  test_access_property_block();

  test_global_implicit_property(hip::access_property::normal{}, cudaAccessProperty::cudaAccessPropertyNormal);
  test_global_implicit_property(hip::access_property::streaming{}, cudaAccessProperty::cudaAccessPropertyStreaming);
  test_global_implicit_property(hip::access_property::persisting{}, cudaAccessProperty::cudaAccessPropertyPersisting);

  test_global();
  test_shared();

  return 0;
}
