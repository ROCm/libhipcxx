//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/random>
// NOTE(HIP/AMD): _CCCL_HOSTED() is 1 under HIPRTC (HIPRTC is not treated as
// freestanding in compiler.h), so gate the host <sstream> explicitly like NVRTC
// to keep it out of the device-only HIPRTC TU.
#if _CCCL_HOSTED()
#  include <sstream>
#endif // _CCCL_HOSTED()

#include "test_macros.h"

template <typename Engine>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_ctor()
{
  Engine e1;
  Engine e2(Engine::default_seed);
  assert(e1 == e2);
  Engine e3(42);
  assert(e3 != e2);
  auto seq = cuda::std::seed_seq{};
  Engine e4(seq);
  Engine e5 = e4;
  assert(e4 == e5);
  static_assert(noexcept(Engine()));
  static_assert(noexcept(Engine(42)));
  return true;
}

template <typename Engine>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_copy()
{
  Engine e1;
  Engine e2 = e1;
  assert(e1 == e2);
  e1();
  assert(e1 != e2);
  e2 = e1;
  assert(e1 == e2);

  static_assert(noexcept(Engine(e1)));
  static_assert(noexcept(e2 = e1));

  return true;
}

template <typename Engine>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_seed()
{
  Engine e1(23);
  Engine e2;
  e2.seed(Engine::default_seed);
  assert(e1 != e2);
  e1.seed(Engine::default_seed);
  assert(e1 == e2);

  auto seq = cuda::std::seed_seq{};
  static_assert(cuda::std::is_void_v<decltype(e1.seed(seq))>);
  static_assert(cuda::std::is_void_v<decltype(e1.seed())>);
  static_assert(cuda::std::is_void_v<decltype(e1.seed(23))>);
  static_assert(noexcept(e1.seed()));
  static_assert(noexcept(e1.seed(23)));
  return true;
}

template <typename Engine>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_operator()
{
  Engine e1;
  static_assert(cuda::std::is_same_v<decltype(e1()), typename Engine::result_type>);
  e1();
  Engine e2;
  assert(e1 != e2);
  e2();
  assert(e1 == e2);
  return true;
}

template <typename Engine, typename Engine::result_type value_10000>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_discard()
{
  Engine e;
  for (int i = 0; i < 100; ++i)
  {
    Engine e2;
    e2.discard(i);
    assert(e == e2);
    e();
  }

  e = Engine();
  e.discard(9999);
  assert(e() == value_10000);

  static_assert(cuda::std::is_void_v<decltype(e.discard(10))>);
  static_assert(noexcept(e.discard(10)));

  return true;
}

template <typename Engine>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_equality()
{
  Engine e;
  assert(e == e);
  Engine e2;
  assert(e == e2);
  e();
  assert(e != e2);
  e  = Engine(3);
  e2 = Engine(3);
  assert(e == e2);
  e2 = Engine(4);
  assert(e != e2);

  static_assert(noexcept(e == e2));
  static_assert(noexcept(e != e2));
  return true;
}

template <typename Engine>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_min_max()
{
  const auto seeds = {0, 29332, 9000};
  for (auto seed : seeds)
  {
    Engine e(seed);
    for (int i = 0; i < 100; ++i)
    {
      auto val = e();
      assert(val <= Engine::max());
      // Avoid pointless comparison of unsigned values with 0 warning
      if constexpr (Engine::min() > 0)
      {
        assert(val >= Engine::min());
      }
    }
  }
  static_assert(Engine::min() <= Engine::max());
  static_assert(noexcept(Engine::min()));
  static_assert(noexcept(Engine::max()));
  static_assert(cuda::std::is_same_v<decltype(Engine::min()), typename Engine::result_type>);
  static_assert(cuda::std::is_same_v<decltype(Engine::max()), typename Engine::result_type>);
  return true;
}

// NOTE(HIP/AMD): gate this host-only test
// (uses std::stringstream) with the explicit HIPRTC exclusion like NVRTC.
#if _CCCL_HOSTED()
template <typename Engine>
void test_save_restore()
{
  Engine e0;
  e0.discard(10000);
  std::stringstream ss;
  ss << e0;

  e0.discard(10000);
  Engine e1;
  ss >> e1;
  e1.discard(10000);
  assert(e0() == e1());
}
#endif // _CCCL_HOSTED()

template <typename Engine, typename Engine::result_type value_10000>
TEST_FUNC TEST_CONSTEXPR_CXX20 bool test_engine()
{
  test_ctor<Engine>();
  test_seed<Engine>();
  test_copy<Engine>();
  test_operator<Engine>();
  test_discard<Engine, value_10000>();
  test_equality<Engine>();
  test_min_max<Engine>();
  NV_IF_TARGET(NV_IS_HOST, ({ test_save_restore<Engine>(); }));
#if TEST_STD_VER >= 2020
  static_assert(test_ctor<Engine>());
  static_assert(test_seed<Engine>());
  static_assert(test_copy<Engine>());
  static_assert(test_operator<Engine>());
  static_assert(test_discard<Engine, value_10000>());
  static_assert(test_equality<Engine>());
  static_assert(test_min_max<Engine>());
#endif
  return true;
}
