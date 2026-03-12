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

# <<<<<<< OLD CODE from f17cf0067f (5c6dd87a64) - COMMENTED OUT
# include(${CMAKE_CURRENT_LIST_DIR}/CCCLC2hHipDeps.cmake)
# =======
cccl_get_cudatoolkit()
# >>>>>>> END NEW CODE (5c6dd87a64)

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.public_headers_host_only)
add_custom_target(libcudacxx.test.public_headers_host_only_with_ctk)

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

# <<<<<<< OLD CODE from bfe97afd5b (49a588ce37) - COMMENTED OUT
# function(libcudacxx_create_public_header_test_host header_name headertest_src)
#   # Create the default target for that file
#   add_library(
#     public_headers_host_only_${header_name}
#     SHARED
#     "${headertest_src}.cpp"
#   )
#   # NOTE(HIP/AMD): tag the configured headertest TU with LANGUAGE HIP so
#   # it goes through the HIP toolchain rather than the plain CXX host
#   # compiler -- this exercises the same compilation pipeline our HIP
#   # consumers use. Lessons-learned ported from the upgrade/3.1_base
#   # test/public_headers_host_only/CMakeLists.txt.
#   if (LIBCUDACXX_ENABLE_HIP)
#     set_source_files_properties(
#       "${headertest_src}.cpp"
#       PROPERTIES LANGUAGE HIP
#     )
#   endif()
#   cccl_configure_target(public_headers_host_only_${header_name})
#   target_include_directories(
#     public_headers_host_only_${header_name}
#     PRIVATE "${libcudacxx_SOURCE_DIR}/include"
#   )
#   target_compile_definitions(
#     public_headers_host_only_${header_name}
#     PRIVATE #
#       ${public_host_header_cxx_compile_definitions}
#       _CCCL_HEADER_TEST
#   )
#   target_compile_options(
#     public_headers_host_only_${header_name}
#     PRIVATE ${public_host_header_cxx_compile_options}
#   )
#   target_link_libraries(
#     public_headers_host_only_${header_name}
#     PUBLIC libcudacxx.compiler_interface
#   )
#   # NOTE(HIP/AMD): under HIP also attach the HIP host runtime + pthreads.
#   # See cmake/CCCLC2hHipDeps.cmake for the manylinux/glibc < 2.34
#   # rationale on Threads::Threads. No-op on non-HIP builds.
#   cccl_c2h_attach_hip_deps(public_headers_host_only_${header_name})
#   add_dependencies(
#     libcudacxx.test.public_headers_host_only
#     public_headers_host_only_${header_name}
#   )
# endfunction()
#
# =======
# >>>>>>> END NEW CODE (49a588ce37)
function(
  libcudacxx_add_public_header_test_host_target
  target_name
  parent_target
  with_ctk
)
  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CXX
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
  if (with_ctk)
    target_link_libraries(${target_name} PUBLIC CUDA::cudart)
  endif()
  add_dependencies(${parent_target} ${target_name})
endfunction()

# <<<<<<< OLD CODE from bfe97afd5b (49a588ce37) - COMMENTED OUT
# function(libcudacxx_add_public_headers_host_only header)
#   # ${header} contains the "/" from the subfolder, replace by "_" for actual names
#   string(REPLACE "/" "_" header_name "${header}")
#
#   # Create the source file for the header target from the template and add the file to the global project
#   set(headertest_src "headers/${header_name}")
#   configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/header_test.cpp.in"
#     "${headertest_src}.cpp"
#   )
#
#   # Create the default target for that file
#   libcudacxx_create_public_header_test_host(${header_name} ${headertest_src})
#   # NOTE(HIP/AMD): the *_with_ctk variant links CUDA::cudart and exercises
#   # CTK-only headers, neither of which exists on a HIP-only build. Skip
#   # it on HIP (the empty libcudacxx.test.public_headers_host_only_with_ctk
#   # umbrella target stays defined near the top so references still resolve).
#   if (NOT LIBCUDACXX_ENABLE_HIP)
#     libcudacxx_create_public_header_test_host_with_ctk(${header_name} ${headertest_src})
#   endif()
# endfunction()
#
# foreach (header IN LISTS public_headers_host_only)
#   libcudacxx_add_public_headers_host_only(${header})
# endforeach()
# =======
libcudacxx_add_public_header_test_host_target(
  libcudacxx.test.public_headers_host_only.base
  libcudacxx.test.public_headers_host_only
  OFF
)
libcudacxx_add_public_header_test_host_target(
  libcudacxx.test.public_headers_host_only_with_ctk.base
  libcudacxx.test.public_headers_host_only_with_ctk
  ON
)
# >>>>>>> END NEW CODE (49a588ce37)
