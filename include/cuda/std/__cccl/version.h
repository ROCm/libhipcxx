//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// This file is somewhat automatically generated. Disable clang-format.
// clang-format off


#ifndef __CCCL_VERSION_H
#define __CCCL_VERSION_H

#define HIPCCL_VERSION 3000002
#define HIPCCL_MAJOR_VERSION (HIPCCL_VERSION / 1000000)
#define HIPCCL_MINOR_VERSION (((HIPCCL_VERSION / 1000) % 1000))
#define HIPCCL_PATCH_VERSION (HIPCCL_VERSION % 1000)

#if HIPCCL_PATCH_VERSION > 99
#  error "HIPCCL patch version cannot be greater than 99 for compatibility with the MMMmmmpp format."
#endif

#endif // __CCCL_VERSION_H
