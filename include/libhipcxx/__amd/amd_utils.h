// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#pragma once
#if defined(__HIP__)
// NOTE(HIP/AMD): hip/hip_runtime.h is not available under HIPRTC, but
// rocm-core/rocm_version.h only contains version macros and extern "C"
// declarations and is safe to include under HIPRTC.
#if !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif
#include <rocm-core/rocm_version.h>
#endif

#ifndef __HIP_DEVICE_COMPILE__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

// NOTE(HIP/AMD): Wave-size selector. AMD wavefronts are 64 lanes wide on
// gfx9 (MI100/MI200/MI300) and 32 lanes wide on RDNA (gfx10+, gfx11, gfx12).
// Used by HIP-side PTX-wrapper emulations of the warp lanemask SREGs to
// derive a wave-size-aware all-lanes mask without leaking bits into
// positions above the actual wave size.
//
// The host pass defaults to wave-32 (no __GFX*__ defined); the bodies
// that consume this macro are __device__-only so this only affects
// codegen in the device pass. clang predefines '__GFX10__' for every
// gfx10xx variant (including the gfx101x range that doesn't get the
// '__GFX10_1__' sub-generation define on older clang releases), so the
// bare '__GFX10__' check covers the whole RDNA1/2 generation.
#if defined(__GFX9__)
#  define _CCCL_HIP_WAVE_SIZE 64
#elif defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__)
#  define _CCCL_HIP_WAVE_SIZE 32
#else
// Unknown/unlisted AMDGPU arch: assume wave-32. Warn (device pass only,
// so host TUs stay quiet) instead of silently assuming, but never error
// -- builds on new archs must not be blocked (cf. the TSC handling in
// <libhipcxx/__amd/hip_tsc_clockrate.h>, ROCm/libhipcxx#22). Set
// _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE to silence it.
#  if defined(__HIP_DEVICE_COMPILE__) && !defined(_LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE) && !defined(NDEBUG)
#    warning Wave size for the current AMDGPU architecture is unknown to libhipcxx; assuming 32 lanes. Warp-level PTX emulations may behave incorrectly. To override this warning, please set the compile-time flag _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE
#  endif
#  define _CCCL_HIP_WAVE_SIZE 32
#endif

// NOTE(HIP/AMD): Managed (hipMallocManaged) memory is fine-grained by
// default on ROCm. Per AMD's GPU atomics support tables
// (https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html,
// "Migratable Host DRAM" -> fine-grained), 32-/64-bit float atomicAdd and
// 64-bit float atomicMin/Max on fine-grained memory are a silent NOP on
// CDNA1/CDNA2 (gfx908 / MI100, gfx90a / MI200) but are natively supported
// from CDNA3 onward (gfx940/941/942 / MI300, gfx950 / MI350) -- both at
// device scope; MI300X/MI350X additionally downgrade (not drop) at system
// scope, and MI300A (APU) is natively correct at system scope too. This
// macro is only defined on architectures where the native instructions are
// safe to use on managed memory; it is undefined (not 0) on gfx908/gfx90a
// and any other/unknown architecture so new targets default to the safe
// (CAS-loop) behavior. Originally lived in test/support/test_macros.h;
// moved here so library code (atomic_hip_derived.h) and the test suite
// share one definition instead of two independently-maintained arch lists.
#ifdef __HIP_PLATFORM_AMD__
#if defined(__GFX9__)
#if !defined(__gfx90a__) and !defined(__gfx906__) and !defined(__gfx908__)
#define LIBHIPCXX_SUPPORTS_MANAGED_MEMORY_ATOMIC_FETCH
#endif
#endif
#endif

namespace libhipcxx
{
  __host__ __device__ inline void __trap(){
    #ifndef NDEBUG
    // #warning "__trap(): the call of __builtin_trap() will abort the host process. \
    // This deviates from the CUDA implementation where __trap() terminates a kernel \
    // and the host process is notified via an error status."
      __builtin_trap();
    #endif
  }
  //===----------------------------------------------------------------------===//
  // Address Space Query Functions
  // Provides CUDA-compatible __isShared and __isGlobal for HIP
  //===----------------------------------------------------------------------===//
  #ifdef __HIP_DEVICE_COMPILE__
  // NOTE(HIP/AMD): the libhipcxx::__is* functions below are unconditional
  // because they live inside namespace libhipcxx and never collide with
  // a future global '::__isShared' (etc.) ROCm may add. The collision
  // risk is at the global-namespace 'using' declarations + host-side
  // stubs further down in this file -- those are guarded by
  // '!defined(<name>) && !__has_builtin(<name>)' so future ROCm
  // releases that ship the same name (either as a regular function
  // declaration or as a clang builtin) automatically suppress our
  // re-export.

  /**
   * @brief Check if a pointer points to shared memory (LDS - Local Data Share)
   * @param ptr Pointer to check (can be any type)
   * @return true if pointer is in shared memory address space
   *
   * This function uses AMD GCN built-in __builtin_amdgcn_is_shared when available
   * (gfx90a and newer). On older architectures, it returns false as a safe fallback.
   */
  template<typename T>
  __device__ inline bool __isShared(const T* ptr) {
#if __has_builtin(__builtin_amdgcn_is_shared)
    // Cast to address space 0 (generic/flat) as required by the builtin
    return __builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)ptr);
#else
    // Fallback for older architectures without the builtin
    (void)ptr;
    return false;
#endif
  }

  /**
   * @brief Check if a pointer points to global memory
   * @param ptr Pointer to check (can be any type)
   * @return true if pointer is in global memory address space
   *
   * In AMD terminology, "global" memory excludes shared (LDS) and private
   * (register/stack) memory. This function checks that the pointer is neither
   * in shared nor private address space.
   */
  template<typename T>
  __device__ inline bool __isGlobal(const T* ptr) {
#if __has_builtin(__builtin_amdgcn_is_shared) && __has_builtin(__builtin_amdgcn_is_private)
    // Cast to address space 0 (generic/flat) as required by the builtin
    const __attribute__((address_space(0))) void* flat_ptr =
        (const __attribute__((address_space(0))) void*)ptr;

    // Global memory is neither shared nor private
    return !__builtin_amdgcn_is_shared(flat_ptr) &&
           !__builtin_amdgcn_is_private(flat_ptr);
#else
    // Fallback: assume it's global if not shared
    // This is a reasonable assumption for pointers allocated on device
    return !__isShared(ptr);
#endif
  }

  /**
   * @brief Check if a pointer points to private memory (registers/stack)
   * @param ptr Pointer to check (can be any type)
   * @return true if pointer is in private memory address space
   *
   * Private memory includes local variables and register-allocated data.
   * This is typically used internally and rarely needed in user code.
   */
  template<typename T>
  __device__ inline bool __isPrivate(const T* ptr) {
#if __has_builtin(__builtin_amdgcn_is_private)
    return __builtin_amdgcn_is_private(
        (const __attribute__((address_space(0))) void*)ptr);
#else
    (void)ptr;
    return false;
#endif
  }

  /**
   * @brief Check if a pointer points to local memory (private/stack)
   * @param ptr Pointer to check (can be any type)
   * @return true if pointer is in local memory address space
   *
   * In AMD/HIP terminology, "local" memory is the same as "private" memory.
   * This is an alias for __isPrivate to match CUDA's __isLocal intrinsic.
   */
  template<typename T>
  __device__ inline bool __isLocal(const T* ptr) {
    return __isPrivate(ptr);
  }

  /**
   * @brief Check if a pointer points to constant memory
   * @param ptr Pointer to check (can be any type)
   * @return false unconditionally; the call also traps the device.
   *
   * NOTE(HIP/AMD): AMDGCN does not expose a `__builtin_amdgcn_is_constant`
   * builtin, and the existing address-space builtins
   * (`__builtin_amdgcn_is_shared`, `__builtin_amdgcn_is_private`) cannot
   * disambiguate `__constant__` memory from regular global memory --
   * `__constant__` globals live in the global address space on AMDGCN
   * and are reported as global by the existing builtins. Rather than
   * silently returning a wrong answer (the previous heuristic had a
   * logic bug that always returned `false`; `is_global` was defined as
   * the same set of conditions as the final return, making them
   * mutually exclusive), trip a hard
   * failure (printf + __builtin_trap) so callers know the query is
   * unsupported. We don't use _CCCL_VERIFY here because including
   * <cuda/std/__cccl/assert.h> from this file (which is pulled in
   * very early via <cuda/std/detail/__config> -> <libhipcxx/__amd/cuda_runtime.h>)
   * baked CCCL_ENABLE_*_ASSERTIONS macros in before lit tests like
   * libcxx/asserts/assert_device_disabled.pass.cpp could '#undef' them.
   * Reintroduce a real implementation once the HIP / AMDGCN toolchain
   * grows a builtin equivalent to NVIDIA's `isspacep.const` PTX op.
   */
  template<typename T>
  __device__ inline bool __isConstant(const T* ptr) {
    (void) ptr;
    printf("ERROR: cuda::device::address_space::constant queries are not "
           "supported on AMDGCN: __isConstant has no AMDGCN builtin "
           "equivalent and __constant__ globals are indistinguishable "
           "from regular global memory.\n");
    __builtin_trap();
    return false;
  }

  /**
   * @brief Check if a pointer points to grid constant memory
   * @param ptr Pointer to check (can be any type)
   * @return false (grid constant not supported on AMD)
   *
   * Grid constant memory is a CUDA-specific feature not available on AMD GPUs.
   */
  template<typename T>
  __device__ inline bool __isGridConstant(const T* ptr) {
    (void)ptr;
    return false;
  }

  /**
   * @brief Check if a pointer points to cluster shared memory
   * @param ptr Pointer to check (can be any type)
   * @return false (cluster shared not supported on AMD)
   *
   * Cluster shared memory is a CUDA compute capability 9.0+ feature not available on AMD GPUs.
   */
  template<typename T>
  __device__ inline bool __isClusterShared(const T* ptr) {
    (void)ptr;
    return false;
  }
  #endif // __HIP_DEVICE_COMPILE__
}

//===----------------------------------------------------------------------===//
// Make address space query functions available in global namespace
// to match CUDA's behavior where __isShared/__isGlobal are global
//===----------------------------------------------------------------------===//
// NOTE(HIP/AMD): each global-scope export below is guarded by the pair
//   !defined(<name>) && !__has_builtin(<name>)
// so a future ROCm release that ships either a regular global
// declaration of '::__isShared' (etc.) or a corresponding clang
// builtin '__isShared' automatically suppresses our re-export -- this
// avoids redefinition errors and using-declaration ambiguities. The
// in-namespace 'libhipcxx::__is*' definitions further up are
// unconditional because they live in a separate namespace and never
// participate in the collision.
#if defined(__HIP__)
  #ifdef __HIP_DEVICE_COMPILE__
#if !defined(__isShared) && !__has_builtin(__isShared)
  using libhipcxx::__isShared;
#endif
#if !defined(__isGlobal) && !__has_builtin(__isGlobal)
  using libhipcxx::__isGlobal;
#endif
#if !defined(__isPrivate) && !__has_builtin(__isPrivate)
  using libhipcxx::__isPrivate;
#endif
#if !defined(__isLocal) && !__has_builtin(__isLocal)
  using libhipcxx::__isLocal;
#endif
#if !defined(__isConstant) && !__has_builtin(__isConstant)
  using libhipcxx::__isConstant;
#endif
#if !defined(__isGridConstant) && !__has_builtin(__isGridConstant)
  using libhipcxx::__isGridConstant;
#endif
#if !defined(__isClusterShared) && !__has_builtin(__isClusterShared)
  using libhipcxx::__isClusterShared;
#endif
  #else
  // Host-side stubs (always return false, since host doesn't have these address spaces)
  // Marked as __host__ __device__ to allow calling from __global__ functions during host compilation
#if !defined(__isShared) && !__has_builtin(__isShared)
  template<typename T>
  __host__ __device__ inline bool __isShared(const T*) { return false; }
#endif
#if !defined(__isGlobal) && !__has_builtin(__isGlobal)
  template<typename T>
  __host__ __device__ inline bool __isGlobal(const T*) { return false; }
#endif
#if !defined(__isPrivate) && !__has_builtin(__isPrivate)
  template<typename T>
  __host__ __device__ inline bool __isPrivate(const T*) { return false; }
#endif
#if !defined(__isLocal) && !__has_builtin(__isLocal)
  template<typename T>
  __host__ __device__ inline bool __isLocal(const T*) { return false; }
#endif
#if !defined(__isConstant) && !__has_builtin(__isConstant)
  template<typename T>
  __host__ __device__ inline bool __isConstant(const T*) { return false; }
#endif
#if !defined(__isGridConstant) && !__has_builtin(__isGridConstant)
  template<typename T>
  __host__ __device__ inline bool __isGridConstant(const T*) { return false; }
#endif
#if !defined(__isClusterShared) && !__has_builtin(__isClusterShared)
  template<typename T>
  __host__ __device__ inline bool __isClusterShared(const T*) { return false; }
#endif
  #endif
#endif

// Returns true if the current ROCm version is at least major.minor.patch
#define LIBHIPCXX_ROCM_VERSION_GE3(major, minor, patch) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     defined(ROCM_VERSION_PATCH) && \
     (ROCM_VERSION_MAJOR > (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR > (minor)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor) && ROCM_VERSION_PATCH >= (patch)))

// Returns true if the current ROCm version is at least major.minor the patch value is ignored
#define LIBHIPCXX_ROCM_VERSION_GE2(major, minor) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     (ROCM_VERSION_MAJOR > (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR >= (minor)))

// Returns true if the current ROCm version is at least major the minor and patch value is ignored
#define LIBHIPCXX_ROCM_VERSION_GE1(major) \
    (defined(ROCM_VERSION_MAJOR) && \
     (ROCM_VERSION_MAJOR >= (major)))

// Returns true if the current ROCm version is at most major.minor.patch (i.e. >=)
#define LIBHIPCXX_ROCM_VERSION_LE3(major, minor, patch) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     defined(ROCM_VERSION_PATCH) && \
     (ROCM_VERSION_MAJOR < (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR < (minor)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor) && ROCM_VERSION_PATCH <= (patch)))

// Returns true if the current ROCm version is at most major.minor (i.e. >=) the patch value is ignored
#define LIBHIPCXX_ROCM_VERSION_LE2(major, minor) \
    (defined(ROCM_VERSION_MAJOR) && \
     defined(ROCM_VERSION_MINOR) && \
     (ROCM_VERSION_MAJOR < (major)) || \
     (ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR <= (minor)))

// Returns true if the current ROCm version is at most major (i.e. >=) the patch and minor value is ignored
#define LIBHIPCXX_ROCM_VERSION_LE1(major) \
    (defined(ROCM_VERSION_MAJOR) && \
     (ROCM_VERSION_MAJOR <= (major)))

// Returns true if the current ROCm version matches exactly major.minor.patch
#define LIBHIPCXX_ROCM_VERSION_EQ3(major, minor, patch) \
     ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor) && ROCM_VERSION_PATCH == (patch)

// Returns true if the current ROCm version matches exactly major.minor (ignoring patch release)
#define LIBHIPCXX_ROCM_VERSION_EQ2(major, minor) \
     ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor)

// Returns true if the current ROCm version matches exactly major (ignoring minor and patch release)
#define LIBHIPCXX_ROCM_VERSION_EQ1(major) \
     ROCM_VERSION_MAJOR == (major) && ROCM_VERSION_MINOR == (minor)

#define LIBHIPCXX_GET_MACRO(_1, _2, _3, name, ...)    name
#define LIBHIPCXX_ROCM_VERSION_GE(...)     LIBHIPCXX_GET_MACRO(__VA_ARGS__, LIBHIPCXX_ROCM_VERSION_GE3, LIBHIPCXX_ROCM_VERSION_GE2, LIBHIPCXX_ROCM_VERSION_GE1)(__VA_ARGS__)

#define LIBHIPCXX_ROCM_VERSION_LE(...)     LIBHIPCXX_GET_MACRO(__VA_ARGS__, LIBHIPCXX_ROCM_VERSION_LE3, LIBHIPCXX_ROCM_VERSION_LE2, LIBHIPCXX_ROCM_VERSION_LE1)(__VA_ARGS__)

#define LIBHIPCXX_ROCM_VERSION_EQ(...)     LIBHIPCXX_GET_MACRO(__VA_ARGS__, LIBHIPCXX_ROCM_VERSION_EQ3, LIBHIPCXX_ROCM_VERSION_EQ2, LIBHIPCXX_ROCM_VERSION_EQ1)(__VA_ARGS__)
