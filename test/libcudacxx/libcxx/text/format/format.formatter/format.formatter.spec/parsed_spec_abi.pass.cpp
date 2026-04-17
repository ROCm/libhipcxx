//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

// <cuda/std/format>

// cuda::std::__fmt_parsed_spec

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstring>
#include <cuda/std/utility>

#include "test_macros.h"

template <class CharT>
struct TestParsedSpecValues
{
  cuda::std::__fmt_spec_std std;
  cuda::std::uint32_t width;
  cuda::std::int32_t precision;
  cuda::std::__fmt_spec_code_point<CharT> fill;
};

template <class CharT>
TEST_FUNC TestParsedSpecValues<CharT> make_test_parsed_spec_values() noexcept
{
  cuda::std::__fmt_spec_std value_std{};
  value_std.__alignment_            = cuda::std::to_underlying(cuda::std::__fmt_spec_alignment::__center);
  value_std.__sign_                 = cuda::std::to_underlying(cuda::std::__fmt_spec_sign::__space);
  value_std.__alternate_form_       = false;
  value_std.__locale_specific_form_ = true;
  value_std.__padding_0_            = false;
  value_std.__type_                 = cuda::std::__fmt_spec_type::__fixed_lower_case;

  TestParsedSpecValues<CharT> value{};
  value.std       = value_std;
  value.width     = 0x1234'5678;
  value.precision = 0x01ab'cdef;
  // value.fill has it's own constructor
  return value;
}

template <class CharT>
TEST_FUNC void verify_parsed_spec(const cuda::std::__fmt_parsed_spec<CharT>& value) noexcept
{
  const auto ref = make_test_parsed_spec_values<CharT>();
  assert(value.__std_.__alignment_ == ref.std.__alignment_);
  assert(value.__std_.__sign_ == ref.std.__sign_);
  assert(value.__std_.__alternate_form_ == ref.std.__alternate_form_);
  assert(value.__std_.__locale_specific_form_ == ref.std.__locale_specific_form_);
  assert(value.__std_.__type_ == ref.std.__type_);
  assert(value.__width_ == ref.width);
  assert(value.__precision_ == ref.precision);
  assert(cuda::std::memcmp(&value.__fill_.__data, &ref.fill.__data, sizeof(ref.fill)) == 0);
}

template <class CharT>
TEST_FUNC void test()
{
  static_assert(sizeof(cuda::std::__fmt_parsed_spec<CharT>) == 16);
  assert(offsetof(cuda::std::__fmt_parsed_spec<CharT>, __std_) == 0);
  assert(offsetof(cuda::std::__fmt_parsed_spec<CharT>, __width_) == 4);
  assert(offsetof(cuda::std::__fmt_parsed_spec<CharT>, __precision_) == 8);
  assert(offsetof(cuda::std::__fmt_parsed_spec<CharT>, __fill_) == 12);

  const auto ref = make_test_parsed_spec_values<CharT>();

  cuda::std::__fmt_parsed_spec<CharT> value{};
  value.__std_       = ref.std;
  value.__width_     = ref.width;
  value.__precision_ = ref.precision;
  value.__fill_      = ref.fill;

  verify_parsed_spec(value);
}

TEST_FUNC void test()
{
  test<char>();
#if _CCCL_HAS_WCHAR_T()
  test<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
template <class CharT>
__global__ void test_host_device_abi_compatiblity_kernel(cuda::std::__fmt_parsed_spec<CharT> value)
{
  verify_parsed_spec(value);
}

template <class CharT>
void test_host_device_abi_compatiblity()
{
  const auto ref = make_test_parsed_spec_values<CharT>();

  cuda::std::__fmt_parsed_spec<CharT> value{};
  value.__std_       = ref.std;
  value.__width_     = ref.width;
  value.__precision_ = ref.precision;
  value.__fill_      = ref.fill;

  test_host_device_abi_compatiblity_kernel<CharT><<<1, 1>>>(value);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

void test_host_device_abi_compatiblity()
{
  test_host_device_abi_compatiblity<char>();
#  if _CCCL_HAS_WCHAR_T()
  test_host_device_abi_compatiblity<wchar_t>();
#  endif // _CCCL_HAS_WCHAR_T()
}
#endif // !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)

int main(int, char**)
{
  test();
  NV_IF_TARGET(NV_IS_HOST, (test_host_device_abi_compatiblity();))
  return 0;
}
