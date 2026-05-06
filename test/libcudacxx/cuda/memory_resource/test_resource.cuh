// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cuda/memory_resource>

#include <cstddef>
#include <cstdint>

// NOTE(HIP/AMD): the upstream-named <cuda_runtime_api.h> does not
// exist on a HIP-only system; libhipcxx's detail/__config pulls
// <amd/cuda_runtime.h> transitively (via <cuda/memory_resource>),
// so this include is unnecessary on HIP. Gate on
// __HIP_PLATFORM_AMD__ (clang-hip preprocessor sets it
// unconditionally) rather than _CCCL_HIP_COMPILATION().
#if !defined(__HIP_PLATFORM_AMD__)
#  include <cuda_runtime_api.h>
#endif // !__HIP_PLATFORM_AMD__
#include <testing.cuh>

using std::size_t;
using std::uintptr_t;

struct Counts
{
  int object_count           = 0;
  int move_count             = 0;
  int copy_count             = 0;
  int allocate_count         = 0;
  int deallocate_count       = 0;
  int allocate_async_count   = 0;
  int deallocate_async_count = 0;
  int equal_to_count         = 0;
  int new_count              = 0;
  int delete_count           = 0;

  friend std::ostream& operator<<(std::ostream& os, const Counts& counts)
  {
    return os
        << "object: " << counts.object_count << ", " //
        << "move: " << counts.move_count << ", " //
        << "copy: " << counts.copy_count << ", " //
        << "allocate_sync: " << counts.allocate_count << ", " //
        << "deallocate_sync: " << counts.deallocate_count << ", " //
        << "allocate: " << counts.allocate_async_count << ", " //
        << "deallocate: " << counts.deallocate_async_count << ", " //
        << "equal_to: " << counts.equal_to_count << ", " //
        << "new: " << counts.new_count << ", " //
        << "delete: " << counts.delete_count;
  }

  friend bool operator==(const Counts& lhs, const Counts& rhs) noexcept
  {
    return lhs.object_count == rhs.object_count && //
           lhs.move_count == rhs.move_count && //
           lhs.copy_count == rhs.copy_count && //
           lhs.allocate_count == rhs.allocate_count && //
           lhs.deallocate_count == rhs.deallocate_count && //
           lhs.allocate_async_count == rhs.allocate_async_count && //
           lhs.deallocate_async_count == rhs.deallocate_async_count && //
           lhs.equal_to_count == rhs.equal_to_count && //
           lhs.new_count == rhs.new_count && //
           lhs.delete_count == rhs.delete_count; //
  }

  friend bool operator!=(const Counts& lhs, const Counts& rhs) noexcept
  {
    return !(lhs == rhs);
  }
};

struct test_fixture_
{
  Counts counts;
  size_t bytes_ = 0;
  size_t align_ = 0;
  static thread_local Counts* counts_;

  test_fixture_() noexcept
      : counts()
  {
    counts_ = &counts;
  }

  size_t bytes(size_t sz) noexcept
  {
    bytes_ = sz;
    return bytes_;
  }

  size_t align(size_t align) noexcept
  {
    align_ = align;
    return align_;
  }
};

inline thread_local Counts* test_fixture_::counts_ = nullptr;

template <class>
using test_fixture = test_fixture_;

struct get_data
{
  using value_type = int;
};

struct extra_property
{};

template <class T>
struct test_resource
{
  int data;
  test_fixture_* fixture;
  T cookie[2] = {0xDEADBEEF, 0xDEADBEEF};

  explicit test_resource(int i, test_fixture_* fix) noexcept
      : data(i)
      , fixture(fix)
  {
    ++fixture->counts.object_count;
  }

  test_resource(test_resource&& other) noexcept
      : data(other.data)
      , fixture(other.fixture)
  {
    other._assert_valid();
    ++fixture->counts.move_count;
    ++fixture->counts.object_count;
    other.cookie[0] = other.cookie[1] = 0x0C07FEFE;
  }

  test_resource(const test_resource& other) noexcept
      : data(other.data)
      , fixture(other.fixture)
  {
    other._assert_valid();
    ++fixture->counts.copy_count;
    ++fixture->counts.object_count;
  }

  ~test_resource()
  {
    --fixture->counts.object_count;
  }

  test_resource& operator=(test_resource other) noexcept
  {
    other._assert_valid();
    ::cuda::std::swap(data, other.data);
    ::cuda::std::swap(fixture, other.fixture);
    return *this;
  }

  void* allocate_sync(std::size_t bytes, std::size_t align)
  {
    _assert_valid();
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.allocate_count;
    return fixture;
  }

  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t align) noexcept
  {
    _assert_valid();
    CHECK(ptr == fixture);
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.deallocate_count;
    return;
  }

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t align)
  {
    _assert_valid();
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.allocate_async_count;
    return fixture;
  }

  void deallocate(::cuda::stream_ref, void* ptr, std::size_t bytes, std::size_t align) noexcept
  {
    _assert_valid();
    CHECK(ptr == fixture);
    CHECK(bytes == fixture->bytes_);
    CHECK(align == fixture->align_);
    ++fixture->counts.deallocate_async_count;
    return;
  }

  friend bool operator==(const test_resource& lhs, const test_resource& rhs)
  {
    lhs._assert_valid();
    rhs._assert_valid();
    ++lhs.fixture->counts.equal_to_count;
    return lhs.data == rhs.data;
  }

  friend bool operator!=(const test_resource& lhs, const test_resource& rhs)
  {
    FAIL("any_resource should only be calling operator==");
    return lhs.data != rhs.data;
  }

  void _assert_valid() const noexcept
  {
    REQUIRE(cookie[0] == 0xDEADBEEF);
    REQUIRE(cookie[1] == 0xDEADBEEF);
  }

  static void* operator new(::cuda::std::size_t size)
  {
    ++test_fixture_::counts_->new_count;
    return ::operator new(size);
  }

  static void operator delete(void* pv) noexcept
  {
    ++test_fixture_::counts_->delete_count;
    return ::operator delete(pv);
  }

  friend constexpr void get_property(const test_resource&, ::cuda::mr::host_accessible) noexcept {}
  friend constexpr void get_property(const test_resource&, extra_property) noexcept {}
  friend constexpr int get_property(const test_resource& self, get_data) noexcept
  {
    return self.data;
  }
};

using big_resource   = test_resource<uintptr_t>;
using small_resource = test_resource<unsigned int>;

static_assert(sizeof(big_resource) > cuda::__default_small_object_size);
static_assert(sizeof(small_resource) <= cuda::__default_small_object_size);
