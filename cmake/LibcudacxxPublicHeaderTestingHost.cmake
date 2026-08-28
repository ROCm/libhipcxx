# MIT License
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

# NOTE(HIP/AMD): upstream 3.4.0 added an unconditional cccl_get_cudatoolkit()
# here. On HIP the macro pulls in a REQUIRED find_package(hiprand); the host
# header sweep does not need the CUDA toolkit at all, so gate it out on HIP
# (mirrors cmake/LibcudacxxInternalHeaderTesting.cmake).
if (NOT LIBCUDACXX_ENABLE_HIP)
  cccl_get_cudatoolkit()
endif()

# NOTE(HIP/AMD): pull in cccl_c2h_attach_hip_deps() so the generated header
# test targets pick up hip::host + Threads::Threads on HIP builds (no-op on
# non-HIP). See cmake/CCCLC2hHipDeps.cmake for the pthreads/glibc rationale.
include(${CMAKE_CURRENT_LIST_DIR}/CCCLC2hHipDeps.cmake)

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers_host_only)
add_custom_target(libcudacxx.test.public_headers_host_only_with_ctk)

if (CCCL_ENABLE_TILE) # TODO(miscco): For now only test public headers with tile
  return()
endif()

# Grep all public headers
file(
  GLOB public_headers_host_only
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/*"
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

# NOTE(HIP/AMD): same block-list pattern as
# cmake/LibcudacxxInternalHeaderTesting.cmake (g17 review on PR #217).
# The public header sweep includes 'cuda/<X>' surfaces (barrier, latch,
# semaphore, ...) that are upstream-only as of 2026; drop them up-front
# so the per-target skip regex inside libcudacxx_create_public_header_test_host
# isn't needed. There is no allow-list here because no public-header
# 'cuda/<X>' file under a SKIP_DIRS prefix currently has a HIP fallback.
if (LIBCUDACXX_ENABLE_HIP)
  include(${CMAKE_CURRENT_LIST_DIR}/LibcudacxxFilterBackendHeaders.cmake)
  libcudacxx_filter_backend_headers(public_headers_host_only
    BACKEND HIP
    SKIP_DIRS
      cuda/barrier cuda/latch cuda/semaphore
      cuda/annotated_ptr cuda/pipeline cuda/memcpy_async cuda/ptx
      cuda/std/barrier cuda/std/latch cuda/std/semaphore
  )
endif()

set(public_host_header_cxx_compile_options)
set(public_host_header_cxx_compile_definitions)

# Specifically add libc++ testing if requested to the libcudacxx host suite
if (CCCL_USE_LIBCXX)
  list(APPEND public_host_header_cxx_compile_options "-stdlib=libc++")
endif()

function(
  libcudacxx_add_public_header_test_host_target
  target_name
  parent_target
  with_ctk
)
  # FIXME(HIP/AMD): on HIP builds tag the host header-test TUs LANGUAGE HIP so the
  # HIP/clang front-end compiles them. The clang-only flags come from hip::device
  # (-x hip, --offload-arch=<gfx>; see hip-config-amd.cmake), not from the
  # hip::host that cccl_c2h_attach_hip_deps() links below -- hip::host carries
  # only -D__HIP_PLATFORM_AMD__=1. hip::device arrives transitively over an
  # all-PUBLIC chain we do not control:
  #   <this target> -> libcudacxx.compiler_interface -> libcudacxx::libcudacxx
  #                 -> CUB / Thrust -> hip::hipcub / roc::rocthrust
  #                 -> roc::rocprim_hip -> hip::device
  # so it cannot be unlinked here. Its options are $<COMPILE_LANGUAGE:CXX>-guarded,
  # which is why LANGUAGE HIP is the fix: the genex goes false and clang drives the
  # compile anyway. With LANGUAGE CXX and a non-clang host compiler
  # (e.g. CMAKE_CXX_COMPILER=g++) they leak onto the CXX compile and fail with
  #   c++: error: unrecognized command-line option '--offload-arch=gfx90a'.
  if (LIBCUDACXX_ENABLE_HIP)
    set(header_lang HIP)
  else()
    set(header_lang CXX)
  endif()
  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE ${header_lang}
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${public_headers_host_only}
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE #
      ${public_host_header_cxx_compile_definitions}
      _CCCL_HEADER_TEST
  )
  target_compile_options(
    ${target_name}
    PRIVATE ${public_host_header_cxx_compile_options}
  )
  target_link_libraries(${target_name} PUBLIC libcudacxx.compiler_interface)
  # NOTE(HIP/AMD): attach the HIP host runtime + pthreads on HIP builds so the
  # generated header-test TUs find <hip/hip_runtime.h> and link std::once_flag.
  # No-op on non-HIP. hip::host itself carries no clang-only flags; the LANGUAGE
  # HIP tagging above is for the hip::device options that reach us transitively.
  cccl_c2h_attach_hip_deps(${target_name})
  if (with_ctk)
    target_link_libraries(${target_name} PUBLIC CUDA::cudart)
  endif()
  add_dependencies(${parent_target} ${target_name})
endfunction()

libcudacxx_add_public_header_test_host_target(
  libcudacxx.test.public_headers_host_only.base
  libcudacxx.test.public_headers_host_only
  OFF
)
# NOTE(HIP/AMD): the *_with_ctk variant links CUDA::cudart and exercises
# CTK-only headers, neither of which exists on a HIP-only build. Skip it on
# HIP; the empty umbrella target defined near the top keeps references valid.
if (NOT LIBCUDACXX_ENABLE_HIP)
  libcudacxx_add_public_header_test_host_target(
    libcudacxx.test.public_headers_host_only_with_ctk.base
    libcudacxx.test.public_headers_host_only_with_ctk
    ON
  )
endif()
