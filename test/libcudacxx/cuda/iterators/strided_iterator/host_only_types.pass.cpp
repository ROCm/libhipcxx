//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "host_device_types.h"
#include "test_macros.h"

void test()
{
  host_only_container vec{};
  {
    using Iter             = typename host_only_container::iterator;
    using strided_iterator = cuda::strided_iterator<Iter, ::std::integral_constant<int, 2>>;

    const strided_iterator default_constructed{};
    strided_iterator value_constructed{vec.begin()};

    strided_iterator copy_constructed{default_constructed};
    strided_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] strided_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] strided_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);

    [[maybe_unused]] strided_iterator iter_stride_constructed{vec.begin(), ::std::integral_constant<int, 2>{}};
  }

  cuda::strided_iterator iter1{vec.begin(), ::std::integral_constant<int, 2>{}};
  const cuda::strided_iterator iter2{vec.begin() + 2, ::std::integral_constant<int, 2>{}};
  assert(iter1 != iter2);

  {
    assert(++iter1 == iter2);
    assert(--iter1 != iter2);
  }

  {
    assert(iter1++ != iter2);
    assert(iter1-- == iter2);
  }

  {
    assert(iter1 + 1 == iter2);
    assert(1 + iter1 == iter2);
    assert(iter1 - 1 != iter2);
    assert(iter2 - iter1 == 1);
  }

  {
    iter1 += 1;
    assert(iter1 == iter2);
    iter1 -= 1;
    assert(iter1 != iter2);
  }

  {
    assert(iter1[1] == 3);
    assert(*iter1 == 1);

    assert(iter2[1] == 5);
    assert(*iter2 == 3);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
