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

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers)

# Grep all public headers
file(
  GLOB public_headers
  LIST_DIRECTORIES false
  RELATIVE "${libcudacxx_SOURCE_DIR}/include"
  CONFIGURE_DEPENDS
  "${libcudacxx_SOURCE_DIR}/include/cuda/*"
  "${libcudacxx_SOURCE_DIR}/include/cuda/std/*"
)

# annotated_ptr does not work with clang cuda due to __nv_associate_access_property
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  list(REMOVE_ITEM public_headers "annotated_ptr")
endif()

# NOTE(HIP/AMD): same block-list pattern as the internal/host-only
# sweeps (PR #217 review). New upstream headers under SKIP_DIRS will
# surface as build failures rather than being silently filtered out.
if (LIBCUDACXX_ENABLE_HIP)
  include(${CMAKE_CURRENT_LIST_DIR}/LibcudacxxFilterBackendHeaders.cmake)
  libcudacxx_filter_backend_headers(public_headers
    BACKEND HIP
    SKIP_DIRS
      cuda/ptx cuda/barrier cuda/latch cuda/semaphore
      cuda/annotated_ptr cuda/pipeline cuda/memcpy_async
      cuda/std/barrier cuda/std/latch cuda/std/semaphore
  )
endif()

function(libcudacxx_add_public_header_test_target target_name)
  if (NOT ARGN)
    return()
  endif()

  # NOTE(HIP/AMD): on HIP, tag the generated .cu files as LANGUAGE HIP so that
  # CMake's HIP toolchain handles them instead of the default CUDA path.
  if (LIBCUDACXX_ENABLE_HIP)
    set(header_lang HIP)
  else()
    set(header_lang CUDA)
  endif()

  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE ${header_lang}
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${ARGN}
  )

  target_compile_definitions(${target_name} PRIVATE _CCCL_HEADER_TEST)
  target_link_libraries(${target_name} PUBLIC libcudacxx.compiler_interface)
  # NOTE(HIP/AMD): also link the HIP device runtime when building for HIP.
  if (LIBCUDACXX_ENABLE_HIP)
    target_link_libraries(${target_name} PUBLIC hip::device)
  endif()
  add_dependencies(libcudacxx.test.public_headers ${target_name})
endfunction()

libcudacxx_add_public_header_test_target(
  libcudacxx.test.public_headers.base
  ${public_headers}
)
