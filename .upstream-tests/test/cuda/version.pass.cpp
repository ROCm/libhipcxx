//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/version>

static_assert(HIPCCL_MAJOR_VERSION == (HIPCCL_VERSION / 1000000), "");
static_assert(HIPCCL_MINOR_VERSION == (HIPCCL_VERSION / 1000 % 1000), "");
static_assert(HIPCCL_PATCH_VERSION == (HIPCCL_VERSION % 1000), "");

// The NVIDIA-branded CCCL_* spellings are gone; make sure they do not creep back.
#ifdef CCCL_VERSION
#  error "CCCL_VERSION must not be defined by libhipcxx; use HIPCCL_VERSION"
#endif

int main(int argc, char** argv)
{
  return 0;
}
