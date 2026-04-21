//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <<<<<<< OLD CODE from 54526887dd (b6b29f65d9) - COMMENTED OUT
// // MIT License
// //
// // Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// //
// // Permission is hereby granted, free of charge, to any person obtaining a copy
// // of this software and associated documentation files (the "Software"), to deal
// // in the Software without restriction, including without limitation the rights
// // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// // copies of the Software, and to permit persons to whom the Software is
// // furnished to do so, subject to the following conditions:
// //
// // The above copyright notice and this permission notice shall be included in all
// // copies or substantial portions of the Software.
// //
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// // SOFTWARE.
// =======
// XFAIL: enable-tile
// error: bit field read/write is unsupported in tile code
// >>>>>>> END NEW CODE (b6b29f65d9)

// <cuda/std/format>

// cuda::std::__fmt_spec_chrono

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct TestSpecChronoValues
{
  cuda::std::__fmt_spec_alignment alignment;
  bool locale_specific_form;
  bool hour;
  bool weekday_name;
  bool weekday;
  bool day_of_year;
  bool week_of_year;
  bool month_name;
};

TEST_FUNC TestSpecChronoValues make_test_spec_chrono_values() noexcept
{
  TestSpecChronoValues value{};
  value.alignment            = cuda::std::__fmt_spec_alignment::__center;
  value.locale_specific_form = false;
  value.hour                 = true;
  value.weekday_name         = true;
  value.weekday              = false;
  value.day_of_year          = false;
  value.week_of_year         = false;
  value.month_name           = true;
  return value;
}

TEST_FUNC void verify_spec_chrono(const cuda::std::__fmt_spec_chrono& value) noexcept
{
  const auto ref = make_test_spec_chrono_values();
  assert(value.__alignment_ == cuda::std::to_underlying(ref.alignment));
  assert(value.__locale_specific_form_ == ref.locale_specific_form);
  assert(value.__hour_ == ref.hour);
  assert(value.__weekday_name_ == ref.weekday_name);
  assert(value.__weekday_ == ref.weekday);
  assert(value.__day_of_year_ == ref.day_of_year);
  assert(value.__week_of_year_ == ref.week_of_year);
  assert(value.__month_name_ == ref.month_name);
}

TEST_FUNC void test()
{
  static_assert(sizeof(cuda::std::__fmt_spec_chrono) == 2);

  const auto ref = make_test_spec_chrono_values();

  cuda::std::__fmt_spec_chrono value{};
  value.__alignment_            = cuda::std::to_underlying(ref.alignment);
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__hour_                 = ref.hour;
  value.__weekday_name_         = ref.weekday_name;
  value.__weekday_              = ref.weekday;
  value.__day_of_year_          = ref.day_of_year;
  value.__week_of_year_         = ref.week_of_year;
  value.__month_name_           = ref.month_name;

  verify_spec_chrono(value);
}

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
__global__ void test_host_device_abi_compatiblity_kernel(const cuda::std::__fmt_spec_chrono value)
{
  verify_spec_chrono(value);
}

void test_host_device_abi_compatiblity()
{
  const auto ref = make_test_spec_chrono_values();

  cuda::std::__fmt_spec_chrono value{};
  value.__alignment_            = cuda::std::to_underlying(ref.alignment);
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__hour_                 = ref.hour;
  value.__weekday_name_         = ref.weekday_name;
  value.__weekday_              = ref.weekday;
  value.__day_of_year_          = ref.day_of_year;
  value.__week_of_year_         = ref.week_of_year;
  value.__month_name_           = ref.month_name;

  test_host_device_abi_compatiblity_kernel<<<1, 1>>>(value);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

int main(int, char**)
{
  test();
  NV_IF_TARGET(NV_IS_HOST, (test_host_device_abi_compatiblity();))
  return 0;
}
