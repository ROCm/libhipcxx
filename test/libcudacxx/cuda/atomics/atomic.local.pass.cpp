//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// UNSUPPORTED: windows && pre-sm-70

// NOTE(HIP/AMD): the test body in main() below is already wrapped in
// '#if !defined(_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE)', so on the
// HIPRTC build (where <cuda/std/__internal/atomic.h> force-defines
// _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE -- see the matching note
// there) main() reduces to 'return 0;' and the test trivially
// passes. No UNSUPPORTED marker is therefore required for HIPRTC,
// and HIPCC keeps the SAFE-path coverage of the actual
// stack-allocated cuda::atomic<T> exercised by tests<...>().

#include <cuda/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename T>
TEST_DEVICE_FUNC T store(T in)
{
  cuda::atomic<T> x(in);
  x.store(in + 1, cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T compare_exchange_weak(T in)
{
  cuda::atomic<T> x(in);
  T old = T(7);
  x.compare_exchange_weak(old, T(42), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T compare_exchange_strong(T in)
{
  cuda::atomic<T> x(in);
  T old = T(7);
  x.compare_exchange_strong(old, T(42), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T exchange(T in)
{
  cuda::atomic<T> x(in);
  T out = x.exchange(T(1), cuda::memory_order_relaxed);
  return out + x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_add(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_add(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_sub(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_sub(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_and(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_and(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_or(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_or(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_xor(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_xor(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_min(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_min(T(7), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC T fetch_max(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_max(T(7), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
TEST_DEVICE_FUNC inline void tests()
{
  const T tid = threadIdx.x;
  assert(tid + T(1) == store(tid));
  assert(T(1) + tid == exchange(tid));
  assert(tid == T(7) ? T(42) : tid == compare_exchange_weak(tid));
  assert(tid == T(7) ? T(42) : tid == compare_exchange_strong(tid));
  assert((tid + T(1)) == fetch_add(tid));
  assert((tid & T(1)) == fetch_and(tid));
  assert((tid | T(1)) == fetch_or(tid));
  assert((tid ^ T(1)) == fetch_xor(tid));
  assert(min(tid, T(7)) == fetch_min(tid));
  assert(max(tid, T(7)) == fetch_max(tid));
  assert(T(tid - T(1)) == fetch_sub(tid));
}

int main(int arg, char** argv)
{
#if !defined(_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE)
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 64;),
    (tests<uint8_t>(); tests<uint16_t>(); tests<uint32_t>(); tests<uint64_t>(); tests<int8_t>(); tests<int16_t>();
     tests<int32_t>();
     tests<int64_t>();))
#endif
  return 0;
}
