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
  set(message_type FATAL_ERROR)
  if (CCCL_ENABLE_CLANG_TIDY)
    # We are here because CI has force-enabled clang-tidy. We must use a debug build for
    # this because certain clang-tidy checks (such as out of bounds or clang static
    # analyzer) work better when they see assert()'s. In this case we don't actually
    # intend to run any of the benchmarks, we just need them to be compilable, so a simple
    # warning is enough.
    #
    # We don't ignore this outright (by making it say, DEBUG or VERBOSE), because it's
    # possible that a user may accidentally stumble into enabling the option.
    set(message_type WARNING)
  endif()
  message(${message_type} "libcu++ benchmarks must be built in release mode.")
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

  cccl_add_executable(${bench_target} NO_METATARGETS SOURCES "${bench_src}")
  if (LIBCUDACXX_ENABLE_HIP)
    set_source_files_properties("${bench_src}" PROPERTIES LANGUAGE HIP)
    # nvbench_helper.cuh's device paths make [[nodiscard]] hip* calls whose returns
    # the upstream CUDA code ignores; don't fail -Werror on that (see nvbench_helper).
    target_compile_options(${bench_target} PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:-Wno-error=unused-value -Wno-error=unused-result>)
  endif()
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

    # NOTE(HIP/AMD): skip benches with no working HIP backend yet (see the list).
    set(_bench_skip FALSE)
    foreach (_skip_pat IN LISTS LIBCUDACXX_BENCH_HIP_SKIP_LIST)
      string(FIND "${bench_name}" "${_skip_pat}" _skip_pos)
      if (NOT _skip_pos EQUAL -1)
        set(_bench_skip TRUE)
        break()
      endif()
    endforeach()
    if (_bench_skip)
      continue()
    endif()

    set(base_bench_name "${bench_name}.base")
    add_bench(base_bench_target ${base_bench_name} "${bench_src}")
    add_dependencies(${benches_meta_target} ${base_bench_target})
    target_compile_definitions(${base_bench_target} PRIVATE TUNE_BASE=1)
    target_compile_options(
      ${base_bench_target}
      PRIVATE "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>"
    )
    # benchmarking
    register_cccl_benchmark("${bench_name}" "")
  endforeach()
endfunction()

# NOTE(HIP/AMD): benchmarks whose PSTL algorithm has no working HIP backend yet.
# Matched as substrings of the dotted bench name (libcudacxx.bench.<algo>.<variant>);
# the surrounding dots keep e.g. `.find.` from matching `find_if`/`.reverse.` from
# matching `reverse_copy`. Re-enable a group when its backend lands.
set(LIBCUDACXX_BENCH_HIP_SKIP_LIST)
if (LIBCUDACXX_ENABLE_HIP)
  set(LIBCUDACXX_BENCH_HIP_SKIP_LIST
    # find_if family: hipCUB has no cub::DeviceFind (see __pstl/cuda/find_if.h),
    # so every algorithm dispatching through it has no selected HIP backend.
    .adjacent_find. .all_of. .any_of. .equal. .find. .find_if. .find_if_not.
    .is_heap. .is_heap_until. .is_partitioned. .is_sorted. .is_sorted_until.
    .mismatch. .none_of.
    # min/max_element: rocPRIM's arg_index_iterator static-asserts the input is a
    # std::random_access_iterator, which the thrust device_vector iterator category
    # (thrust::random_access_device_iterator_tag) does not satisfy.
    .max_element. .min_element.
    # iter_swap / multi-input transform materialization write through rocThrust
    # proxy references (device_reference), which are not yet fully swappable/
    # assignable in these paths (single-input transform.fib DOES build).
    .reverse. .swap_ranges. .transform.babelstream
    # backends not ported to hipCUB/rocPRIM yet (DeviceSelect::Unique signature,
    # ForEachN-policy stable_partition).
    .unique. .unique_copy. .stable_partition.
  )
endif()

get_recursive_subdirs(subdirs)

foreach (subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
