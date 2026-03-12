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

cccl_get_cudatoolkit()

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.internal_headers)

# <<<<<<< OLD CODE from bfe97afd5b (49a588ce37) - COMMENTED OUT
# # <<<<<<< OLD CODE from f17cf0067f (5c6dd87a64) - COMMENTED OUT
# # if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
# #   find_package(NVHPC)
# # elseif (NOT LIBCUDACXX_ENABLE_HIP)
# #   # NOTE(HIP/AMD): on HIP there is no CUDA Toolkit; cccl_get_cudatoolkit()
# #   # does a REQUIRED find_package(CUDAToolkit) that hard-fails. The HIP
# #   # internal header tests link hip::device (see cudart_name below) instead.
# #   cccl_get_cudatoolkit()
# # endif()
# #
# # =======
# # >>>>>>> END NEW CODE (5c6dd87a64)
# # We need to handle atomic headers differently as they do not compile on architectures below sm70
# set(architectures_at_least_sm70)
# foreach (item IN LISTS CMAKE_CUDA_ARCHITECTURES)
#   if (item GREATER_EQUAL 70)
#     list(APPEND architectures_at_least_sm70 ${item})
#   endif()
# endforeach()
#
# =======
# >>>>>>> END NEW CODE (49a588ce37)
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

# <<<<<<< OLD CODE from bfe97afd5b (49a588ce37) - COMMENTED OUT
# # <<<<<<< OLD CODE from f17cf0067f (5c6dd87a64) - COMMENTED OUT
# # # NOTE(HIP/AMD): under HIP, filter out the upstream feature surfaces that
# # # have no HIP-portable implementation as of 2026, then re-add the
# # # individual cuda::ptx::* wrappers that DO ship a HIP software emulation
# # # (see the consolidated NOTE in <cuda/__ptx/ptx_helper_functions.h>) so
# # # the emulations don't bitrot. Block-list / allow-list pattern (per the
# # # g17 review feedback on PR #217) -- adding a new upstream header under
# # # one of the SKIP_DIRS will surface as a build failure rather than being
# # # silently skipped, which makes loss-of-coverage regressions visible.
# # if (LIBCUDACXX_ENABLE_HIP)
# #   include(${CMAKE_CURRENT_LIST_DIR}/LibcudacxxFilterBackendHeaders.cmake)
# #   libcudacxx_filter_backend_headers(internal_headers
# #     BACKEND HIP
# #     SKIP_DIRS
# #       cuda/__barrier cuda/__latch cuda/__semaphore
# #       cuda/__annotated_ptr cuda/__pipeline cuda/__memcpy_async cuda/__ptx
# #       # CUDA-only PTX atomic dispatch headers (inline-PTX asm; the HIP
# #       # path uses atomic_hip_{generated,derived}.h instead).
# #       cuda/std/__atomic/functions/cuda_ptx_generated.h
# #       cuda/std/__atomic/functions/cuda_ptx_derived.h
# #       cuda/std/__atomic/functions/cuda_ptx_generated_helper.h
# #     ALLOWLIST_HEADERS
# #       cuda/__ptx/ptx_helper_functions.h
# #       cuda/__ptx/ptx_dot_variants.h
# #       cuda/__ptx/instructions/bmsk.h
# #       cuda/__ptx/instructions/elect_sync.h
# #       cuda/__ptx/instructions/fence.h
# #       cuda/__ptx/instructions/get_sreg.h
# #       cuda/__ptx/instructions/shfl_sync.h
# #       cuda/__ptx/instructions/shl.h
# #       cuda/__ptx/instructions/shr.h
# #       cuda/__ptx/instructions/trap.h
# #   )
# #   set(cudart_name hip::device)
# # elseif ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
# #   set(cudart_name NVHPC::CUDART)
# # else()
# #   set(cudart_name CUDA::cudart)
# # endif()
# #
# # =======
# # >>>>>>> END NEW CODE (5c6dd87a64)
# function(libcudacxx_create_internal_header_test header_name headertest_src)
#   # Create the default target for that file. The TU is always written to
#   # disk as '.cu' (see configure_file in libcudacxx_add_internal_header_test
#   # below); on HIP we override CMake's default '.cu -> LANGUAGE CUDA'
#   # association by tagging the source file with LANGUAGE HIP, which routes
#   # it through the HIP toolchain. clang's HIP front-end accepts '.cu' files
#   # natively, so no separate '.cpp' shadow file is needed.
#   add_library(internal_headertest_${header_name} SHARED "${headertest_src}.cu")
#   cccl_configure_target(internal_headertest_${header_name})
#   if (LIBCUDACXX_ENABLE_HIP)
#     set_source_files_properties(
#       "${headertest_src}.cu"
#       PROPERTIES LANGUAGE HIP
#     )
#   endif()
#   target_include_directories(
#     internal_headertest_${header_name}
#     PRIVATE "${libcudacxx_SOURCE_DIR}/include"
#   )
#   target_compile_definitions(
#     internal_headertest_${header_name}
#     PRIVATE _CCCL_HEADER_TEST
#   )
#   # Bring in the global CCCL compile definitions
#   # Link against the right runtime
# =======
function(libcudacxx_add_internal_header_test_target target_name)
  if (NOT ARGN)
    return()
  endif()

  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CUDA
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${ARGN}
  )

  target_compile_definitions(${target_name} PRIVATE _CCCL_HEADER_TEST)
# >>>>>>> END NEW CODE (49a588ce37)
  target_link_libraries(
    ${target_name}
    PUBLIC #
      libcudacxx.compiler_interface
      CUDA::cudart
  )
# <<<<<<< OLD CODE from bfe97afd5b (49a588ce37) - COMMENTED OUT
#
#   # Ensure that if this is an atomic header, we only include the right architectures
#   # (CUDA-only; the HIP path skipped these via the regex above).
#   if (NOT LIBCUDACXX_ENABLE_HIP)
#     string(
#       REGEX MATCH
#       "atomic|barrier|latch|semaphore|annotated_ptr|pipeline"
#       match
#       "${header}"
#     )
#     if (match)
#       # Ensure that we only compile the header when we have some architectures enabled
#       if (NOT architectures_at_least_sm70)
#         return()
#       endif()
#       set_target_properties(
#         internal_headertest_${header_name}
#         PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}"
#       )
#     endif()
#   endif()
#
#   add_dependencies(
#     libcudacxx.test.internal_headers
#     internal_headertest_${header_name}
#   )
# =======
  add_dependencies(libcudacxx.test.internal_headers ${target_name})
# >>>>>>> END NEW CODE (49a588ce37)
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

# <<<<<<< OLD CODE from bfe97afd5b (49a588ce37) - COMMENTED OUT
#   # Create the source file for the header target from the template. The TU is
#   # always written as '.cu' regardless of backend; on HIP the
#   # libcudacxx_create_internal_header_test helper overrides CMake's default
#   # '.cu -> LANGUAGE CUDA' association via set_source_files_properties.
#   set(headertest_src "headers/${header_name}")
#   configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in"
#     "${headertest_src}.cu"
# =======
if (internal_headers_fallback)
  cccl_generate_header_tests(
    libcudacxx.test.internal_headers.fallback
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CUDA
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${internal_headers_fallback}
    PER_HEADER_DEFINES ${internal_headers_fallback_per_header_defines}
# >>>>>>> END NEW CODE (49a588ce37)
  )
  target_compile_definitions(
    libcudacxx.test.internal_headers.fallback
    PRIVATE _CCCL_HEADER_TEST
  )
  target_link_libraries(
    libcudacxx.test.internal_headers.fallback
    PUBLIC #
      libcudacxx.compiler_interface
      CUDA::cudart
  )
  add_dependencies(
    libcudacxx.test.internal_headers
    libcudacxx.test.internal_headers.fallback
  )
endif()
