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

# This file defines the `libcudacxx_build_compiler_targets()` function, which
# creates the following interface targets:
#
# libcudacxx.compiler_interface
# - Interface target linked into all targets in the libcudacxx developer build.
#   Defines common warning flags, definitions, etc, including those defined in
#   the global CCCL targets.

cccl_get_libcudacxx()

function(libcudacxx_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  #  if (CCCL_USE_LIBCXX)
  #    list(APPEND cxx_compile_options "-stdlib=libc++")
  #    list(APPEND cxx_compile_definitions "_ALLOW_UNSUPPORTED_LIBCPP=1")
  #  endif()

  # Set test specific flags
  list(APPEND cxx_compile_definitions "CCCL_ENABLE_ASSERTIONS")
  list(APPEND cxx_compile_definitions "CCCL_IGNORE_DEPRECATED_CPP_DIALECT")
  list(
    APPEND cxx_compile_definitions
    "CCCL_IGNORE_DEPRECATED_DISCARD_MEMORY_HEADER"
  )
  list(
    APPEND cxx_compile_definitions
    "CCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER"
  )

  if (CCCL_ENABLE_TILE)
    list(APPEND cuda_compile_options "--enable-tile")
  endif()

  cccl_build_compiler_interface(
    libcudacxx.compiler_flags
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  add_library(libcudacxx.compiler_interface INTERFACE)
  if (LIBCUDACXX_ENABLE_EXPERIMENTAL_HIPCUB_SHIM)
    target_compile_definitions(
      libcudacxx.compiler_interface
      INTERFACE LIBHIPCXX_ENABLE_EXPERIMENTAL_HIPCUB_SHIM
    )
  endif()
  target_link_libraries(
    libcudacxx.compiler_interface
    INTERFACE
      # order matters here, we need the libcudacxx options to override the cccl options.
      cccl.compiler_interface
      libcudacxx.compiler_flags
      libcudacxx::libcudacxx
  )
endfunction()
