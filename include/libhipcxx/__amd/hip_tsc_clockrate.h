//===----------------------------------------------------------------------===//
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//===----------------------------------------------------------------------===//

// Per-architecture HIP wall_clock64() TSC clock rate. Consumed by
// libhipcxx code that converts between HIP TSC cycles and seconds /
// nanoseconds (currently <cuda/std/ctime> and
// <cuda/std/__thread/threading_support_cuda.h>). Only active during
// the HIP device pass (__HIP_DEVICE_COMPILE__).
//
// Architecture matrix:
//   __gfx908__/__gfx90a__         -> 25 MHz  (40 ns/cycle)
//   __gfx940/941/942/950__        -> 100 MHz (10 ns/cycle)
//   other __GFX9__                -> 100 MHz, with NDEBUG warning
//   __GFX10__                     -> 100 MHz, with NDEBUG warning
//   __GFX11__                     -> 100 MHz, with NDEBUG warning (RDNA3 ISA)
//   __GFX12__                     -> 100 MHz, with NDEBUG warning (RDNA4 ISA)
//   unknown                       -> NDEBUG warning (silenced by
//                                    _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE);
//                                    macros default to 1 to make the
//                                    resulting timings obviously wrong.

#ifndef _AMD_HIP_TSC_CLOCKRATE_H
#define _AMD_HIP_TSC_CLOCKRATE_H

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
// gfx90a / gfx908: 25 MHz TSC for wall_clock64 -> 1e9/25e6 = 40 ns/cycle
#  if defined(__gfx908__) || defined(__gfx90a__)
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             25000000
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 40
#    endif
// gfx940 / gfx941 / gfx942 / gfx950: 100 MHz TSC -> 1e9/100e6 = 10 ns/cycle
#  elif defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             100000000
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 10
#    endif
// other gfx9: assume 100 MHz TSC.
#  elif defined(__GFX9__)
#    ifndef NDEBUG
#      warning Assuming 100 MHz realtime clock rate (TSC) for this gfx9 architecture. Timing-related APIs (e.g., chrono) or sleep instructions may behave incorrectly!
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             100000000
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 10
#    endif
// gfx1030 / RDNA2: not specified in ISA docs.
#  elif defined(__GFX10__)
#    ifndef NDEBUG
#      warning Assuming 100 MHz realtime clock rate (TSC) for gfx1030. Timing-related APIs (e.g., chrono) or sleep instructions may behave incorrectly!
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             100000000
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 10
#    endif
// gfx1100 / gfx1101: RDNA3 ISA states "constantly running clock (typically 100MHz)".
#  elif defined(__GFX11__)
#    ifndef NDEBUG
#      warning Assuming 100 MHz realtime clock rate (TSC) for gfx1100/gfx1101 (according to the RDNA3 ISA). Timing-related APIs (e.g., chrono) or sleep instructions may behave incorrectly!
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             100000000
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 10
#    endif
// gfx1200 / gfx1201: RDNA4 ISA, assumed 100 MHz.
#  elif defined(__GFX12__)
#    ifndef NDEBUG
#      warning Assuming 100 MHz realtime clock rate (TSC) for gfx1200/gfx1201 (from the RDNA4 ISA). Timing-related APIs (e.g., chrono) or sleep instructions may behave incorrectly!
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             100000000
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 10
#    endif
#  else
// Only a warning (not an error) so builds on new/unlisted archs (e.g.
// gfx13xx) are not blocked -- see ROCm/libhipcxx#22. Set
// _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE to silence it.
#    ifndef _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE
#      ifndef NDEBUG
#        warning Timing-related utility APIs (e.g., chrono) are currently not supported on the current architecture by libhipcxx. To override this warning, please set the compile-time flag _LIBCUDACXX_ALLOW_UNSUPPORTED_ARCHITECTURE
#      endif
#    endif
// Intentionally meaningless values to make the resulting timings clearly wrong.
#    ifndef _LIBCUDACXX_HIP_TSC_CLOCKRATE
#      define _LIBCUDACXX_HIP_TSC_CLOCKRATE             1
#    endif
#    ifndef _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE
#      define _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE 1
#    endif
#  endif
#endif // defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)

#endif // _AMD_HIP_TSC_CLOCKRATE_H
