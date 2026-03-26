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

include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

cccl_get_nvbench_helper()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "libcu++ benchmarks must be built in release mode.")
endif()

if (NOT LIBCUDACXX_ENABLE_HIP AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(
    FATAL_ERROR
    "CMAKE_CUDA_ARCHITECTURES must be set to build libcu++ benchmarks."
  )
endif()
# NOTE(HIP/AMD): on HIP the bench .cu files are compiled as HIP for
# CMAKE_HIP_ARCHITECTURES (set per add_bench below).
if (LIBCUDACXX_ENABLE_HIP AND NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_HIP_ARCHITECTURES must be set to build libcu++ benchmarks on HIP.")
endif()

set(benches_meta_target libcudacxx.all.benches)
add_custom_target(${benches_meta_target})

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(
    GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach (test_dir IN LISTS contents)
    if (IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

create_benchmark_registry()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  if (LIBCUDACXX_ENABLE_HIP)
    set_source_files_properties("${bench_src}" PROPERTIES LANGUAGE HIP)
    # nvbench_helper.cuh's device paths make [[nodiscard]] hip* calls whose returns
    # the upstream CUDA code ignores; don't fail -Werror on that (see nvbench_helper).
    target_compile_options(${bench_target} PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-error=unused-value -Wno-error=unused-result>)
  endif()
  cccl_configure_target(${bench_target} DIALECT 17)
  target_link_libraries(
    ${bench_target}
    PRIVATE libcudacxx::libcudacxx cccl.nvbench_helper nvbench::main
  )
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach (bench_src IN LISTS bench_srcs)
    # base tuning
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "libcudacxx.${bench_prefix}.")

    set(base_bench_name "${bench_name}.base")
    add_bench(base_bench_target ${base_bench_name} "${bench_src}")
    add_dependencies(${benches_meta_target} ${base_bench_target})
    target_compile_definitions(${base_bench_target} PRIVATE TUNE_BASE=1)
# <<<<<<< OLD CODE from 2eddf6d094 (b1db6ffd85) - COMMENTED OUT
#     # --extended-lambda is an nvcc-only flag; clang-HIP supports device lambdas natively.
#     if (NOT LIBCUDACXX_ENABLE_HIP)
#       target_compile_options(${base_bench_target} PRIVATE "--extended-lambda")
#     endif()
# =======
    target_compile_options(
      ${base_bench_target}
      PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
    )
# >>>>>>> END NEW CODE (b1db6ffd85)
    # benchmarking
    register_cccl_benchmark("${bench_name}" "")
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach (subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
