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
  list(FILTER public_headers EXCLUDE REGEX "annotated_ptr")
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

# We need to handle atomic headers differently as they do not compile on architectures below sm70
set(architectures_at_least_sm70)
foreach (item IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if (item GREATER_EQUAL 70)
    list(APPEND architectures_at_least_sm70 ${item})
  endif()
endforeach()

function(libcudacxx_create_public_header_test header_name headertest_src)
# <<<<<<< OLD CODE from 23063816f0 (3fd8e38e4b) - COMMENTED OUT
#   # Create the default target for that file. The TU is always written to
#   # disk as '.cu' (see configure_file in libcudacxx_add_public_header_test
#   # below); on HIP we override CMake's default '.cu -> LANGUAGE CUDA'
#   # association by tagging the source file with LANGUAGE HIP, which routes
#   # it through the HIP toolchain. clang's HIP front-end accepts '.cu' files
#   # natively, so no separate '.cpp' shadow file is needed.
#   set(public_headertest_${header_name} verify_${header_name})
#   add_library(public_headertest_${header_name} SHARED "${headertest_src}.cu")
#   if (LIBCUDACXX_ENABLE_HIP)
#     set_source_files_properties(
#       "${headertest_src}.cu"
#       PROPERTIES LANGUAGE HIP
#     )
#   endif()
# =======
  # Create the default target for that file
  add_library(public_headertest_${header_name} SHARED "${headertest_src}.cu")
  cccl_configure_target(public_headertest_${header_name})
# <<<<<<< OLD CODE from 8c38351f70 (59ad1b152f) - COMMENTED OUT
# # >>>>>>> END NEW CODE (3fd8e38e4b)
#   target_include_directories(
#     public_headertest_${header_name}
#     PRIVATE "${libcudacxx_SOURCE_DIR}/include"
#   )
# =======
# >>>>>>> END NEW CODE (59ad1b152f)
  target_compile_definitions(
    public_headertest_${header_name}
    PRIVATE _CCCL_HEADER_TEST
  )

  # Bring in the global CCCL compile definitions
  target_link_libraries(
    public_headertest_${header_name}
    PUBLIC libcudacxx.compiler_interface
  )
  # NOTE(HIP/AMD): under HIP, also link the HIP device runtime.
  if (LIBCUDACXX_ENABLE_HIP)
    target_link_libraries(
      public_headertest_${header_name}
      PUBLIC hip::device
    )
  endif()

  # Ensure that if this is an atomic header, we only include the right architectures
  # (CUDA-only; the HIP path skipped these via the regex above).
  if (NOT LIBCUDACXX_ENABLE_HIP)
    string(
      REGEX MATCH
      "atomic|barrier|latch|semaphore|annotated_ptr|pipeline"
      match
      "${header}"
    )
    if (match)
      # Ensure that we only compile the header when we have some architectures enabled
      if (NOT architectures_at_least_sm70)
        return()
      endif()
      set_target_properties(
        public_headertest_${header_name}
        PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}"
      )
    endif()
  endif()

  add_dependencies(
    libcudacxx.test.public_headers
    public_headertest_${header_name}
  )
endfunction()

function(libcudacxx_add_public_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template. The TU
  # is always written as '.cu' regardless of backend; on HIP the
  # libcudacxx_create_public_header_test helper overrides CMake's default
  # '.cu -> LANGUAGE CUDA' association via set_source_files_properties.
  set(headertest_src "headers/${header_name}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in"
    "${headertest_src}.cu"
  )

  # Create the default target for that file
  libcudacxx_create_public_header_test(${header_name} ${headertest_src})
endfunction()

foreach (header IN LISTS public_headers)
  libcudacxx_add_public_header_test(${header})
endforeach()
