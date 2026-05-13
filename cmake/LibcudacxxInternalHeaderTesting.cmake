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
add_custom_target(libcudacxx.test.internal_headers)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  find_package(NVHPC)
else()
  find_package(CUDAToolkit)
endif()

# We need to handle atomic headers differently as they do not compile on architectures below sm70
set(architectures_at_least_sm70)
foreach (item IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if (item GREATER_EQUAL 70)
    list(APPEND architectures_at_least_sm70 ${item})
  endif()
endforeach()

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

function(libcudacxx_create_internal_header_test header_name headertest_src)
  # NOTE(HIP/AMD): Skip headers without HIP support as of 2026 when
  # building under HIP -- with one carve-out: individual cuda::ptx::*
  # wrapper headers that ship a HIP software emulation (see the
  # consolidated NOTE in <cuda/__ptx/ptx_helper_functions.h>) are still
  # tested standalone so the emulations don't bitrot. Their pure-C++
  # helper headers (ptx_helper_functions.h, ptx_dot_variants.h) are
  # also tested.
  if (LIBCUDACXX_ENABLE_HIP)
    string(
      REGEX MATCH
      "ptx|barrier|latch|semaphore|annotated_ptr|pipeline|memcpy_async"
      match
      "${header_name}"
    )
    if (match)
      # NOTE(HIP/AMD): header_name has had '/' replaced with '_' by the
      # caller (libcudacxx_add_internal_header_test, line 188) so the
      # carve-out regex works on the underscore form, e.g.
      # 'cuda___ptx_instructions_shl.h'.
      string(
        REGEX MATCH
        "__ptx_(ptx_helper_functions|ptx_dot_variants)\\.h$|__ptx_instructions_(bmsk|elect_sync|fence|get_sreg|shfl_sync|shl|shr|trap)\\.h$"
        hip_emul_match
        "${header_name}"
      )
      if (NOT hip_emul_match)
        return()
      endif()
    endif()
  endif()

  # Create the default target for that file. The TU is always written to
  # disk as '.cu' (see configure_file in libcudacxx_add_internal_header_test
  # below); on HIP we override CMake's default '.cu -> LANGUAGE CUDA'
  # association by tagging the source file with LANGUAGE HIP, which routes
  # it through the HIP toolchain. clang's HIP front-end accepts '.cu' files
  # natively, so no separate '.cpp' shadow file is needed.
  set(internal_headertest_${header_name} verify_${header_name})
  add_library(internal_headertest_${header_name} SHARED "${headertest_src}.cu")
  if (LIBCUDACXX_ENABLE_HIP)
    set_source_files_properties(
      "${headertest_src}.cu"
      PROPERTIES LANGUAGE HIP
    )
  endif()
  target_include_directories(
    internal_headertest_${header_name}
    PRIVATE "${libcudacxx_SOURCE_DIR}/include"
  )
  target_compile_definitions(
    internal_headertest_${header_name}
    PRIVATE _CCCL_HEADER_TEST
  )
  cccl_configure_target(
    internal_headertest_${header_name}
    DIALECT ${CMAKE_CUDA_STANDARD}
  )

  # Bring in the global CCCL compile definitions
  # Link against the right runtime
  if (LIBCUDACXX_ENABLE_HIP)
    target_link_libraries(
      internal_headertest_${header_name}
      PUBLIC #
        libcudacxx.compiler_interface
        hip::device
    )
  elseif ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    target_link_libraries(
      internal_headertest_${header_name}
      PUBLIC #
        libcudacxx.compiler_interface
        NVHPC::CUDART
    )
  else()
    target_link_libraries(
      internal_headertest_${header_name}
      PUBLIC #
        libcudacxx.compiler_interface
        CUDA::cudart
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
        internal_headertest_${header_name}
        PROPERTIES CUDA_ARCHITECTURES "${architectures_at_least_sm70}"
      )
    endif()
  endif()

  add_dependencies(
    libcudacxx.test.internal_headers
    internal_headertest_${header_name}
  )
endfunction()

# We have fallbacks for some type traits that we want to explicitly test so that they do not bitrot
function(
  libcudacxx_create_internal_header_fallback_test
  header_name
  headertest_src
)
  # MSVC cannot handle some of the fallbacks
  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (
      "${header}" MATCHES "is_base_of"
      OR "${header}" MATCHES "is_nothrow_destructible"
      OR "${header}" MATCHES "is_polymorphic"
    )
      return()
    endif()
  endif()

  # Search the file for a fallback definition
  file(READ ${libcudacxx_SOURCE_DIR}/include/${header} header_file)
  string(REGEX MATCH "_LIBCUDACXX_[A-Z_]*_FALLBACK" fallback "${header_file}")
  if (fallback)
    # Adopt the filename for the fallback tests
    set(header_name "${header_name}_fallback")
    libcudacxx_create_internal_header_test(${header_name} ${headertest_src})
    target_compile_definitions(
      internal_headertest_${header_name}
      PRIVATE "-D${fallback}"
    )
  endif()
endfunction()

function(libcudacxx_add_internal_header_test header)
  # ${header} contains the "/" from the subfolder, replace by "_" for actual names
  string(REPLACE "/" "_" header_name "${header}")

  # Create the source file for the header target from the template. The TU is
  # always written as '.cu' regardless of backend; on HIP the
  # libcudacxx_create_internal_header_test helper overrides CMake's default
  # '.cu -> LANGUAGE CUDA' association via set_source_files_properties.
  set(headertest_src "headers/${header_name}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in"
    "${headertest_src}.cu"
  )

  # Create the default target for that file
  libcudacxx_create_internal_header_test(${header_name} ${headertest_src})

  # Optionally create a fallback target for that file
  libcudacxx_create_internal_header_fallback_test(${header_name} ${headertest_src})
endfunction()

foreach (header IN LISTS internal_headers)
  libcudacxx_add_internal_header_test(${header})
endforeach()
