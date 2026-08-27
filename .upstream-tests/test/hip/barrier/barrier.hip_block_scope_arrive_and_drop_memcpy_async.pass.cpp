// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: pre-gfx1250
// UNSUPPORTED: nvcc, nvhpc, nvc++

// <cuda/barrier>

// Its intent is simple: prove tx-active arrive_and_drop() remains correct when
// the tx dependency is produced by ordinary cuda::memcpy_async(..., barrier)
// instead of the manual tx-completion test helper.

#include <cuda/barrier>
#include <cuda/std/chrono>
#include <cuda/std/utility>
#include "test_macros.h"
#include "hip_wavefront_size.h"

constexpr int k_count = 256;

__device__ __attribute__((aligned(16))) int g_src[k_count] = {
  300, 301, 302, 303, 304, 305, 306, 307,
  308, 309, 310, 311, 312, 313, 314, 315,
  316, 317, 318, 319, 320, 321, 322, 323,
  324, 325, 326, 327, 328, 329, 330, 331,
  332, 333, 334, 335, 336, 337, 338, 339,
  340, 341, 342, 343, 344, 345, 346, 347,
  348, 349, 350, 351, 352, 353, 354, 355,
  356, 357, 358, 359, 360, 361, 362, 363,
  364, 365, 366, 367, 368, 369, 370, 371,
  372, 373, 374, 375, 376, 377, 378, 379,
  380, 381, 382, 383, 384, 385, 386, 387,
  388, 389, 390, 391, 392, 393, 394, 395,
  396, 397, 398, 399, 400, 401, 402, 403,
  404, 405, 406, 407, 408, 409, 410, 411,
  412, 413, 414, 415, 416, 417, 418, 419,
  420, 421, 422, 423, 424, 425, 426, 427,
  428, 429, 430, 431, 432, 433, 434, 435,
  436, 437, 438, 439, 440, 441, 442, 443,
  444, 445, 446, 447, 448, 449, 450, 451,
  452, 453, 454, 455, 456, 457, 458, 459,
  460, 461, 462, 463, 464, 465, 466, 467,
  468, 469, 470, 471, 472, 473, 474, 475,
  476, 477, 478, 479, 480, 481, 482, 483,
  484, 485, 486, 487, 488, 489, 490, 491,
  492, 493, 494, 495, 496, 497, 498, 499,
  500, 501, 502, 503, 504, 505, 506, 507,
  508, 509, 510, 511, 512, 513, 514, 515,
  516, 517, 518, 519, 520, 521, 522, 523,
  524, 525, 526, 527, 528, 529, 530, 531,
  532, 533, 534, 535, 536, 537, 538, 539,
  540, 541, 542, 543, 544, 545, 546, 547,
  548, 549, 550, 551, 552, 553, 554, 555
};

__device__ int test()
{
  int const wave_size = get_wavefront_size();
  bool const is_drop_wave = threadIdx.x < wave_size;
  int const expected = 2 * wave_size;

  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  __shared__ __attribute__((aligned(16))) int dest[k_count];
  __shared__ int result;

  cuda::barrier<cuda::thread_scope_block>::arrival_token token = {};
  cuda::barrier<cuda::thread_scope_block>::arrival_token reuse_token = {};

  if (threadIdx.x == 0) {
    init(&bar, expected);
    for (int index = 0; index < k_count; ++index) {
      dest[index] = -1;
    }
    result = 0;

    cuda::memcpy_async(
      &dest[0],
      &g_src[0],
      cuda::aligned_size_t<16>(sizeof(int) * k_count),
      bar);
  }
  __syncthreads();

  if (is_drop_wave) {
    bar.arrive_and_drop();
  } else {
    token = bar.arrive();
    bar.wait(cuda::std::move(token));

    for (int index = threadIdx.x - wave_size; index < k_count; index += wave_size) {
      if (dest[index] != g_src[index]) {
        __atomic_store_n(&result, 1, __ATOMIC_RELEASE);
      }
    }
  }

  __syncthreads();

  if (!is_drop_wave && __atomic_load_n(&result, __ATOMIC_ACQUIRE) == 0) {
    reuse_token = bar.arrive();
  }

  __syncthreads();

  if (!is_drop_wave && __atomic_load_n(&result, __ATOMIC_ACQUIRE) == 0) {
    if (!bar.try_wait_for(cuda::std::move(reuse_token), cuda::std::chrono::nanoseconds(0))) {
      __atomic_store_n(&result, 2, __ATOMIC_RELEASE);
    }
  }

  __syncthreads();
  return __atomic_load_n(&result, __ATOMIC_ACQUIRE);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (
    int const wave_size = get_wavefront_size();
    cuda_block_count = 1;
    cuda_thread_count = 2 * wave_size;
  ))
  NV_IF_TARGET(NV_IS_DEVICE, (return test();))
  return 0;
}