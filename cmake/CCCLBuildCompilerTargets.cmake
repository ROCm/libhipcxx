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

# This file defines the `cccl_build_compiler_targets()` function, which
# creates the following interface targets:
#
# cccl.compiler_interface
# - Interface target providing compiler-specific options needed to build
#   CCCL's tests, examples, etc. for the current CMAKE_CUDA_STANDARD.
#   This includes warning flags and the like.

# sccache cannot handle the -Fd option generating pdb files
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)

option(CCCL_ENABLE_EXCEPTIONS "Enable exceptions within CCCL libraries." ON)
option(CCCL_ENABLE_RTTI "Enable RTTI within CCCL libraries." ON)
option(CCCL_ENABLE_WERROR "Treat warnings as errors for CCCL targets." ON)
option(
  CCCL_ENABLE_PRAGMA_SYSTEM_HEADER
  "When OFF, disables the system header pragma in CCCL headers so that their warnings are visible."
  OFF
)
option(CCCL_ENABLE_PTXAS_WARNINGS "Enable ptxas warnings" OFF) # currently used only in CUB

function(
  cccl_build_compiler_interface
  interface_target
  cuda_compile_options
  cxx_compile_options
  compile_defs
)
  # We test to see if C++ compiler options exist using try-compiles in the CXX lang, and then reuse those flags as
  # -Xcompiler flags for CUDA targets. This requires that the CXX compiler and CUDA_HOST compilers are the same when
  # using nvcc.
  if (CCCL_TOPLEVEL_PROJECT AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(cuda_host_matches_cxx_compiler FALSE)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
      set(
        host_info
        "${CMAKE_CUDA_HOST_COMPILER} (${CMAKE_CUDA_HOST_COMPILER_ID} ${CMAKE_CUDA_HOST_COMPILER_VERSION})"
      )
      set(
        cxx_info
        "${CMAKE_CXX_COMPILER} (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})"
      )
      if (
        CMAKE_CUDA_HOST_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID
        AND
          CMAKE_CUDA_HOST_COMPILER_VERSION
            VERSION_EQUAL
            CMAKE_CXX_COMPILER_VERSION
      )
        set(cuda_host_matches_cxx_compiler TRUE)
      endif()
    else() # CMake < 3.31 doesn't have the CMAKE_CUDA_HOST_COMPILER_ID/VERSION variables
      set(host_info "${CMAKE_CUDA_HOST_COMPILER}")
      set(cxx_info "${CMAKE_CXX_COMPILER}")
      if (CMAKE_CUDA_HOST_COMPILER STREQUAL CMAKE_CXX_COMPILER)
        set(cuda_host_matches_cxx_compiler TRUE)
      endif()
    endif()

    if (NOT cuda_host_matches_cxx_compiler)
      message(
        FATAL_ERROR
        "CCCL developer builds require that CMAKE_CUDA_HOST_COMPILER matches "
        "CMAKE_CXX_COMPILER when using nvcc:\n"
        "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}\n"
        "CMAKE_CUDA_HOST_COMPILER: ${host_info}\n"
        "CMAKE_CXX_COMPILER: ${cxx_info}\n"
        "Rerun cmake with \"-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX_COMPILER}\".\n"
        "Alternatively, configure the CUDAHOSTCXX and CXX environment variables to match.\n"
      )
    endif()
  endif()

  add_library(${interface_target} INTERFACE)

  foreach (cuda_option IN LISTS cuda_compile_options)
    target_compile_options(
      ${interface_target}
      INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${cuda_option}>
    )
  endforeach()

  # NOTE(HIP/AMD): we also mirror the cxx warning/diagnostic flags onto
  # COMPILE_LANGUAGE:HIP so the libcudacxx / libhipcxx HIP test surface (header
  # tests, c2h tests) is built with -Wall / -Wextra / -Werror too -- otherwise
  # regressions caught on the NV side land silently on the HIP side, invisible
  # until someone injects 'CMAKE_HIP_FLAGS' and the latent warnings flood out.
  #
  # BUT the cxx flags were validated against CMAKE_CXX_COMPILER via
  # append_option_if_available() -> check_cxx_compiler_flag(). That host compiler
  # may be GCC while the HIP compiler is clang, so GCC-only flags (e.g.
  # -Walloc-zero, -Wtsan) pass the cxx check yet make `clang -x hip` fail under
  # -Werror,-Wunknown-warning-option. So only mirror a flag to HIP when the HIP
  # compiler itself accepts it (check_compiler_flag detects the "unknown warning
  # option" diagnostic and returns false).
  if (LIBCUDACXX_ENABLE_HIP)
    include(CheckCompilerFlag)
  endif()
  foreach (cxx_option IN LISTS cxx_compile_options)
    target_compile_options(
      ${interface_target}
      INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_option}>
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=${cxx_option}>
    )
    if (LIBCUDACXX_ENABLE_HIP)
      string(MAKE_C_IDENTIFIER "HIP_FLAG_${cxx_option}" _cccl_hip_flag_var)
      check_compiler_flag(HIP "${cxx_option}" ${_cccl_hip_flag_var})
      if (${_cccl_hip_flag_var})
        target_compile_options(
          ${interface_target}
          INTERFACE $<$<COMPILE_LANGUAGE:HIP>:${cxx_option}>
        )
      endif()
    endif()
  endforeach()

  # NOTE(HIP/AMD): HIP-specific warning relaxations, emitted AFTER the mirrored
  # cxx warning flags above so they take precedence on the HIP command line.
  #  - -Wno-nvcc-compat: upstream enables -Wnvcc-compat, which flags every
  #    __host__/__device__ target-attribute function overload as an NVCC
  #    incompatibility. Such overloads are valid and pervasive under clang's HIP
  #    front-end, so the warning is meaningless here (and fatal under -Werror).
  #  - -Wno-error=unused-value / -Wno-error=unused-result: HIP runtime APIs
  #    return [[nodiscard]] hipError_t; library/device paths legitimately ignore
  #    some returns. Matches the existing c2h / benchmark HIP WAR.
  target_compile_options(
    ${interface_target}
    INTERFACE
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-nvcc-compat>
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-error=unused-value>
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-error=unused-result>
  )

  target_compile_definitions(${interface_target} INTERFACE ${compile_defs})
endfunction()

function(cccl_build_compiler_targets)
  set(cuda_compile_options)
  set(cxx_compile_options)
  set(cxx_compile_definitions)

  list(APPEND cuda_compile_options "-Xcudafe=--display_error_number")
  list(APPEND cuda_compile_options "-Wno-deprecated-gpu-targets")
  if (CCCL_ENABLE_WERROR)
    list(APPEND cuda_compile_options "-Xcudafe=--promote_warnings")
  endif()

  if (NOT CCCL_ENABLE_PRAGMA_SYSTEM_HEADER)
    # Ensure that we build our tests without treating ourself as system header
    list(APPEND cxx_compile_definitions "_CCCL_NO_SYSTEM_HEADER")
  endif()

  if (NOT CCCL_ENABLE_EXCEPTIONS)
    list(APPEND cxx_compile_definitions "CCCL_DISABLE_EXCEPTIONS")
  endif()

  if (NOT CCCL_ENABLE_RTTI)
    list(APPEND cxx_compile_definitions "CCCL_DISABLE_RTTI")
  endif()

  #  if (CCCL_USE_LIBCXX)
  #    list(APPEND cxx_compile_options "-stdlib=libc++")
  #    list(APPEND cxx_compile_definitions "_ALLOW_UNSUPPORTED_LIBCPP=1")
  #  endif()

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    list(APPEND cuda_compile_options "--use-local-env")
    list(APPEND cxx_compile_options "/bigobj")
    list(APPEND cxx_compile_definitions "_ENABLE_EXTENDED_ALIGNED_STORAGE")
    list(APPEND cxx_compile_definitions "NOMINMAX")

    append_option_if_available("/W4" cxx_compile_options)
    # Treat all warnings as errors. This is only supported on Release builds,
    # as `nv_exec_check_disable` doesn't seem to work with MSVC debug iterators
    # and spurious warnings are emitted.
    # See NVIDIA/thrust#1273, NVBug 3129879.
    if (CCCL_ENABLE_WERROR)
      if (CMAKE_BUILD_TYPE STREQUAL "Release")
        append_option_if_available("/WX" cxx_compile_options)
      endif()
    endif()

    # Suppress overly-pedantic/unavoidable warnings brought in with /W4:
    # C4324: structure was padded due to alignment specifier
    append_option_if_available("/wd4324" cxx_compile_options)
    # C4505: unreferenced local function has been removed
    # The CUDA `host_runtime.h` header emits this for
    # `__cudaUnregisterBinaryUtil`.
    append_option_if_available("/wd4505" cxx_compile_options)
    # C4706: assignment within conditional expression
    # MSVC doesn't provide an opt-out for this warning when the assignment is
    # intentional. Clang will warn for these, but suppresses the warning when
    # double-parentheses are used around the assignment. We'll let Clang catch
    # unintentional assignments and suppress all such warnings on MSVC.
    append_option_if_available("/wd4706" cxx_compile_options)

    # MSVC STL assumes that `allocator_traits`'s allocator will use raw pointers,
    # and the `__DECLSPEC_ALLOCATOR` macro causes issues with thrust's universal
    # allocators:
    #   warning C4494: 'std::allocator_traits<_Alloc>::allocate' :
    #      Ignoring __declspec(allocator) because the function return type is not
    #      a pointer or reference
    # See https://github.com/microsoft/STL/issues/696
    append_option_if_available("/wd4494" cxx_compile_options)

    # Get error messages with a little arrow indicating the error location more exactly
    append_option_if_available("/diagnostics:caret" cxx_compile_options)

    if (MSVC_TOOLSET_VERSION LESS 143)
      # winbase.h(9572): warning C5105: macro expansion producing 'defined' has undefined behavior
      append_option_if_available("/wd5105" cxx_compile_options)
    endif()
  else()
    list(APPEND cuda_compile_options "-Wreorder")

    if (CCCL_ENABLE_WERROR)
      append_option_if_available("-Werror" cxx_compile_options)
    endif()

    append_option_if_available("-Wall" cxx_compile_options)
    append_option_if_available("-Wextra" cxx_compile_options)
    append_option_if_available("-Wreorder" cxx_compile_options)
    append_option_if_available("-Winit-self" cxx_compile_options)
    append_option_if_available("-Woverloaded-virtual" cxx_compile_options)
    append_option_if_available("-Wcast-qual" cxx_compile_options)
    append_option_if_available("-Wpointer-arith" cxx_compile_options)
    append_option_if_available("-Wunused-local-typedefs" cxx_compile_options)
    append_option_if_available("-Wvla" cxx_compile_options)

    # Clang-only
    append_option_if_available("-Wnvcc-compat" cxx_compile_options)
    append_option_if_available("-Wimplicit-fallthrough" cxx_compile_options)
    append_option_if_available(
      "-fdiagnostics-show-template-tree"
      cxx_compile_options
    )
    append_option_if_available("-Wignored-qualifiers" cxx_compile_options)
    append_option_if_available(
      "-Wmissing-field-initializers"
      cxx_compile_options
    )
    # Inundated with error: ISO C++11 requires at least one argument for the "..." in a
    # variadic macro for _CCCL_REQUIRES_EXPR(), so cannot enable this.
    #
    # append_option_if_available("-pedantic" cxx_compile_options)
    append_option_if_available("-Wsign-compare" cxx_compile_options)
    append_option_if_available(
      "-Warray-bounds-pointer-arithmetic"
      cxx_compile_options
    )
    append_option_if_available("-Wassign-enum" cxx_compile_options)
    append_option_if_available("-Wformat-pedantic" cxx_compile_options)
    append_option_if_available("-Walloc-size" cxx_compile_options)
    append_option_if_available("-Walloc-zero" cxx_compile_options)
    append_option_if_available("-Wtsan" cxx_compile_options)
    append_option_if_available("-Wenum-conversion" cxx_compile_options)
    append_option_if_available("-Wpacked" cxx_compile_options)
    # Clang and GCC
    append_option_if_available(
      "-ftemplate-backtrace-limit=0"
      cxx_compile_options
    )
    append_option_if_available("-fmacro-backtrace-limit=0" cxx_compile_options)
    # Disable GNU extensions (flag is clang only)
    append_option_if_available("-Wgnu" cxx_compile_options)
    append_option_if_available("-Wno-gnu-line-marker" cxx_compile_options) # WAR 3916341
    # Calling a variadic macro with zero args is a GNU extension until C++20,
    # but the THRUST_PP_ARITY macro is used with zero args. Need to see if this
    # is a real problem worth fixing.
    append_option_if_available(
      "-Wno-gnu-zero-variadic-macro-arguments"
      cxx_compile_options
    )

    # This complains about functions in CUDA system headers when used with nvcc.
    append_option_if_available("-Wno-unused-function" cxx_compile_options)
  endif()

  if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
      # GCC 7.3 complains about name mangling changes due to `noexcept`
      # becoming part of the type system; we don't care.
      append_option_if_available("-Wno-noexcept-type" cxx_compile_options)
    endif()
  endif()

  cccl_build_compiler_interface(
    cccl.compiler_interface
    "${cuda_compile_options}"
    "${cxx_compile_options}"
    "${cxx_compile_definitions}"
  )

  # Clang-cuda only:
  target_compile_options(
    cccl.compiler_interface
    INTERFACE
      $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Xclang=-fcuda-allow-variadic-functions>
      $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:-Wno-unknown-cuda-version>
  )
endfunction()
