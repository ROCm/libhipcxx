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

# NOTE(HIP/AMD): on HIP there is no CUDA Toolkit; cccl_get_cudatoolkit()
# does a REQUIRED find_package(CUDAToolkit) that hard-fails. The HIP
# internal header tests link hip::device instead.
# TODO(HIP/AMD): upstream 3.4.0 calls cccl_get_cudatoolkit() unconditionally;
# gate it here to avoid hard-fail on HIP builds.
if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  find_package(NVHPC)
elseif (NOT LIBCUDACXX_ENABLE_HIP)
  cccl_get_cudatoolkit()
endif()

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.internal_headers)

# Grep all internal headers
file(
  GLOB_RECURSE internal_headers
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  ${libcudacxx_SOURCE_DIR}/include/cuda/__*/*.h
  ${libcudacxx_SOURCE_DIR}/include/cuda/std/__*/*.h
)

# Exclude <cuda/std/__cccl/(prologue|epilogue|visibility).h> from the test
list(
  FILTER internal_headers
  EXCLUDE
  REGEX "__cccl/(prologue|epilogue|visibility)\.h"
)

# headers in `__cuda` are meant to come after the related "cuda" headers so they do not compile on their own
list(FILTER internal_headers EXCLUDE REGEX "__cuda/*")

# generated cuda::ptx headers are not standalone
list(FILTER internal_headers EXCLUDE REGEX "__ptx/instructions/generated")

# NOTE(HIP/AMD): under HIP, filter out the upstream feature surfaces that
# have no HIP-portable implementation as of 2026, then re-add the
# individual cuda::ptx::* wrappers that DO ship a HIP software emulation
# (see the consolidated NOTE in <cuda/__ptx/ptx_helper_functions.h>) so
# the emulations don't bitrot. Block-list / allow-list pattern (per the
# g17 review feedback on PR #217) -- adding a new upstream header under
# one of the SKIP_DIRS will surface as a build failure rather than being
# silently skipped, which makes loss-of-coverage regressions visible.
if (LIBCUDACXX_ENABLE_HIP)
  include(${CMAKE_CURRENT_LIST_DIR}/LibcudacxxFilterBackendHeaders.cmake)
  libcudacxx_filter_backend_headers(internal_headers
    BACKEND HIP
    SKIP_DIRS
      cuda/__barrier cuda/__latch cuda/__semaphore
      cuda/__annotated_ptr cuda/__pipeline cuda/__memcpy_async cuda/__ptx
      # CUDA-only PTX atomic dispatch headers (inline-PTX asm; the HIP
      # path uses atomic_hip_{generated,derived}.h instead).
      cuda/std/__atomic/functions/cuda_ptx_generated.h
      cuda/std/__atomic/functions/cuda_ptx_derived.h
      cuda/std/__atomic/functions/cuda_ptx_generated_helper.h
    ALLOWLIST_HEADERS
      cuda/__ptx/ptx_helper_functions.h
      cuda/__ptx/ptx_dot_variants.h
      cuda/__ptx/instructions/bmsk.h
      cuda/__ptx/instructions/elect_sync.h
      cuda/__ptx/instructions/fence.h
      cuda/__ptx/instructions/get_sreg.h
      cuda/__ptx/instructions/shfl_sync.h
      cuda/__ptx/instructions/shl.h
      cuda/__ptx/instructions/shr.h
      cuda/__ptx/instructions/trap.h
  )
  set(cudart_name hip::device)
elseif ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  set(cudart_name NVHPC::CUDART)
else()
  set(cudart_name CUDA::cudart)
endif()

function(libcudacxx_add_internal_header_test_target target_name)
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

  # NOTE(HIP/AMD): compile the pstl backend headers even when the experimental
  # hipCUB shim is off by default, so turning it off does not silently drop
  # cuda/std/__pstl/cuda/* from header-test coverage (they would become empty TUs).
  target_compile_definitions(
    ${target_name}
    PRIVATE #
      _CCCL_HEADER_TEST
      $<$<BOOL:${LIBCUDACXX_ENABLE_HIP}>:LIBHIPCXX_ENABLE_EXPERIMENTAL_HIPCUB_SHIM>
  )
  target_link_libraries(
    ${target_name}
    PUBLIC #
      libcudacxx.compiler_interface
      ${cudart_name}
  )
  add_dependencies(libcudacxx.test.internal_headers ${target_name})
endfunction()

libcudacxx_add_internal_header_test_target(
  libcudacxx.test.internal_headers.base
  ${internal_headers}
)

# We have fallbacks for some type traits that we want to explicitly test so that they do not bitrot.
set(internal_headers_fallback)
set(internal_headers_fallback_per_header_defines)
foreach (header IN LISTS internal_headers)
  # MSVC cannot handle some of the fallbacks.
  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (
      "${header}" MATCHES "is_base_of"
      OR "${header}" MATCHES "is_nothrow_destructible"
      OR "${header}" MATCHES "is_polymorphic"
    )
      continue()
    endif()
  endif()

  file(READ "${libcudacxx_SOURCE_DIR}/include/${header}" header_file)
  string(REGEX MATCH "_LIBCUDACXX_[A-Z_]*_FALLBACK" fallback "${header_file}")
  if (fallback)
    list(APPEND internal_headers_fallback "${header}")
    string(
      REGEX REPLACE
      "([][+.*^$()|?\\\\])"
      "\\\\\\1"
      header_regex
      "${header}"
    )
    list(
      APPEND internal_headers_fallback_per_header_defines
      DEFINE
      "${fallback}"
      "^${header_regex}$"
    )
  endif()
endforeach()

if (internal_headers_fallback)
  # NOTE(HIP/AMD): mirror the backend language/runtime dispatch used by the base
  # target so fallback header tests also route through the HIP toolchain and link
  # the HIP device runtime (cudart_name is set at directory scope above).
  if (LIBCUDACXX_ENABLE_HIP)
    set(header_lang HIP)
  else()
    set(header_lang CUDA)
  endif()
  cccl_generate_header_tests(
    libcudacxx.test.internal_headers.fallback
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE ${header_lang}
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${internal_headers_fallback}
    PER_HEADER_DEFINES ${internal_headers_fallback_per_header_defines}
  )
  target_compile_definitions(
    libcudacxx.test.internal_headers.fallback
    PRIVATE _CCCL_HEADER_TEST
  )
  target_link_libraries(
    libcudacxx.test.internal_headers.fallback
    PUBLIC #
      libcudacxx.compiler_interface
      ${cudart_name}
  )
  add_dependencies(
    libcudacxx.test.internal_headers
    libcudacxx.test.internal_headers.fallback
  )
endif()
