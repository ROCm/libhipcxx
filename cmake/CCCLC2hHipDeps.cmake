# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# cccl_c2h_attach_hip_deps(target [SCOPE PUBLIC|INTERFACE|PRIVATE])
#
# Attach the HIP-side runtime deps that every HIP test target needs:
#   * hip::host        -- HIP runtime API headers / link
#   * Threads::Threads -- '-pthread' link line; required for the
#                         std::once_flag inside cuda::__physical_device
#                         (see the matching block comment in
#                         c2h/CMakeLists.txt around the manylinux /
#                         glibc < 2.34 weak-stub WAR for the rationale).
#
# Default SCOPE is PUBLIC. On non-HIP builds the call is a no-op so the
# helper can be sprinkled unconditionally in shared CMake code (e.g. the
# public/internal header-testing modules that run for both CUDA and
# HIP builds).
#
# Self-installing: find_package(hip) / find_package(Threads) are run
# transparently if the imported targets aren't already present, so the
# helper works whether or not the parent CMakeLists set them up.
function(cccl_c2h_attach_hip_deps target)
  if (NOT LIBCUDACXX_ENABLE_HIP)
    return()
  endif()

  set(opts "")
  set(one  "SCOPE")
  set(many "")
  cmake_parse_arguments(arg "${opts}" "${one}" "${many}" ${ARGN})
  if (NOT arg_SCOPE)
    set(arg_SCOPE PUBLIC)
  endif()

  if (NOT TARGET hip::host)
    find_package(hip REQUIRED)
  endif()
  if (NOT TARGET Threads::Threads)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
  endif()

  target_link_libraries(${target} ${arg_SCOPE} hip::host Threads::Threads)
endfunction()
