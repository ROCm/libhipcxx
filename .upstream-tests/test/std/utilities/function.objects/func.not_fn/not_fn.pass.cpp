//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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

// UNSUPPORTED: c++98, c++03, c++11, c++14

// XFAIL: pgi
// TODO: there's multiple failures that appear to be all about overload resolution and SFINAE,
// and they will require further debugging to pinpoint the root cause of (almost certainly a
// compiler bug)

// template <class F> unspecified not_fn(F&& f);

#include <hip/std/functional>
#include <hip/std/type_traits>
// #include <hip/std/string>
#include <hip/std/cassert>

#include "test_macros.h"
// #include "type_id.h"

#pragma nv_diag_suppress set_but_not_used

///////////////////////////////////////////////////////////////////////////////
//                       CALLABLE TEST TYPES
///////////////////////////////////////////////////////////////////////////////

__host__ __device__
bool returns_true() { return true; }

template <class Ret = bool>
struct MoveOnlyCallable {
  MoveOnlyCallable(MoveOnlyCallable const&) = delete;
  __host__ __device__
  MoveOnlyCallable(MoveOnlyCallable&& other)
      : value(other.value)
  { other.value = !other.value; }

  template <class ...Args>
  __host__ __device__
  Ret operator()(Args&&...) { return Ret{value}; }

  __host__ __device__
  explicit MoveOnlyCallable(bool x) : value(x) {}
  Ret value;
};

template <class Ret = bool>
struct CopyCallable {
  __host__ __device__
  CopyCallable(CopyCallable const& other)
      : value(other.value) {}

  __host__ __device__
  CopyCallable(CopyCallable&& other)
      : value(other.value) { other.value = !other.value; }

  template <class ...Args>
  __host__ __device__
  Ret operator()(Args&&...) { return Ret{value}; }

  __host__ __device__
  explicit CopyCallable(bool x) : value(x)  {}
  Ret value;
};


template <class Ret = bool>
struct ConstCallable {
  __host__ __device__
  ConstCallable(ConstCallable const& other)
      : value(other.value) {}

  __host__ __device__
  ConstCallable(ConstCallable&& other)
      : value(other.value) { other.value = !other.value; }

  template <class ...Args>
  __host__ __device__
  Ret operator()(Args&&...) const { return Ret{value}; }

  __host__ __device__
  explicit ConstCallable(bool x) : value(x)  {}
  Ret value;
};



template <class Ret = bool>
struct NoExceptCallable {
  __host__ __device__
  NoExceptCallable(NoExceptCallable const& other)
      : value(other.value) {}

  template <class ...Args>
  __host__ __device__
  Ret operator()(Args&&...) noexcept { return Ret{value}; }

  template <class ...Args>
  __host__ __device__
  Ret operator()(Args&&...) const noexcept { return Ret{value}; }

  __host__ __device__
  explicit NoExceptCallable(bool x) : value(x)  {}
  Ret value;
};

struct CopyAssignableWrapper {
  CopyAssignableWrapper(CopyAssignableWrapper const&) = default;
  CopyAssignableWrapper(CopyAssignableWrapper&&) = default;
  CopyAssignableWrapper& operator=(CopyAssignableWrapper const&) = default;
  CopyAssignableWrapper& operator=(CopyAssignableWrapper &&) = default;

  template <class ...Args>
  __host__ __device__
  bool operator()(Args&&...) { return value; }

  __host__ __device__
  explicit CopyAssignableWrapper(bool x) : value(x) {}
  bool value;
};


struct MoveAssignableWrapper {
  MoveAssignableWrapper(MoveAssignableWrapper const&) = delete;
  MoveAssignableWrapper(MoveAssignableWrapper&&) = default;
  MoveAssignableWrapper& operator=(MoveAssignableWrapper const&) = delete;
  MoveAssignableWrapper& operator=(MoveAssignableWrapper &&) = default;

  template <class ...Args>
  __host__ __device__
  bool operator()(Args&&...) { return value; }

  __host__ __device__
  explicit MoveAssignableWrapper(bool x) : value(x) {}
  bool value;
};

struct MemFunCallable {
  __host__ __device__
  explicit MemFunCallable(bool x) : value(x) {}

  __host__ __device__
  bool return_value() const { return value; }
  __host__ __device__
  bool return_value_nc() { return value; }
  bool value;
};

enum CallType : unsigned {
  CT_None,
  CT_NonConst = 1,
  CT_Const = 2,
  CT_LValue = 4,
  CT_RValue = 8
};

  __host__ __device__
inline constexpr CallType operator|(CallType LHS, CallType RHS) {
    return static_cast<CallType>(static_cast<unsigned>(LHS) | static_cast<unsigned>(RHS));
}

#if 0

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
__device__
#endif
CallType      ForwardingCallObject_last_call_type = CT_None;
#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
__device__
#endif
TypeID const* ForwardingCallObject_last_call_args = nullptr;

struct ForwardingCallObject {

  template <class ...Args>
  __host__ __device__
  bool operator()(Args&&...) & {
      set_call<Args&&...>(CT_NonConst | CT_LValue);
      return true;
  }

  template <class ...Args>
  __host__ __device__
  bool operator()(Args&&...) const & {
      set_call<Args&&...>(CT_Const | CT_LValue);
      return true;
  }

  // Don't allow the call operator to be invoked as an rvalue.
  template <class ...Args>
  __host__ __device__
  bool operator()(Args&&...) && {
      set_call<Args&&...>(CT_NonConst | CT_RValue);
      return true;
  }

  template <class ...Args>
  __host__ __device__
  bool operator()(Args&&...) const && {
      set_call<Args&&...>(CT_Const | CT_RValue);
      return true;
  }

  template <class ...Args>
  __host__ __device__
  static void set_call(CallType type) {
      assert(ForwardingCallObject_last_call_type == CT_None);
      assert(ForwardingCallObject_last_call_args == nullptr);
      ForwardingCallObject_last_call_type = type;
      ForwardingCallObject_last_call_args = &makeArgumentID<Args...>();
  }

  template <class ...Args>
  __host__ __device__
  static bool check_call(CallType type) {
      bool result =
           ForwardingCallObject_last_call_type == type
        && ForwardingCallObject_last_call_args
        && *ForwardingCallObject_last_call_args == makeArgumentID<Args...>();
      ForwardingCallObject_last_call_type = CT_None;
      ForwardingCallObject_last_call_args = nullptr;
      return result;
  }
};

#endif


///////////////////////////////////////////////////////////////////////////////
//                        BOOL TEST TYPES
///////////////////////////////////////////////////////////////////////////////

#if (defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__))
__device__
#endif
int EvilBool_bang_called = 0;

struct EvilBool {
  EvilBool(EvilBool const&) = default;
  EvilBool(EvilBool&&) = default;

  __host__ __device__
  friend EvilBool operator!(EvilBool const& other) {
    ++EvilBool_bang_called;
    return EvilBool{!other.value};
  }

private:
  friend struct MoveOnlyCallable<EvilBool>;
  friend struct CopyCallable<EvilBool>;
  friend struct NoExceptCallable<EvilBool>;

  __host__ __device__
  explicit EvilBool(bool x) : value(x) {}
  EvilBool& operator=(EvilBool const& other) = default;

public:
  bool value;
};

struct ExplicitBool {
  ExplicitBool(ExplicitBool const&) = default;
  ExplicitBool(ExplicitBool&&) = default;

  __host__ __device__
  explicit operator bool() const { return value; }

private:
  friend struct MoveOnlyCallable<ExplicitBool>;
  friend struct CopyCallable<ExplicitBool>;

  __host__ __device__
  explicit ExplicitBool(bool x) : value(x) {}
  __host__ __device__
  ExplicitBool& operator=(bool x) {
      value = x;
      return *this;
  }

  bool value;
};


struct NoExceptEvilBool {
  NoExceptEvilBool(NoExceptEvilBool const&) = default;
  NoExceptEvilBool(NoExceptEvilBool&&) = default;
  NoExceptEvilBool& operator=(NoExceptEvilBool const& other) = default;

  __host__ __device__
  explicit NoExceptEvilBool(bool x) : value(x) {}

  __host__ __device__
  friend NoExceptEvilBool operator!(NoExceptEvilBool const& other) noexcept {
    return NoExceptEvilBool{!other.value};
  }

  bool value;
};



__host__ __device__
void constructor_tests()
{
    {
        using T = MoveOnlyCallable<bool>;
        T value(true);
        using RetT = decltype(hip::std::not_fn(hip::std::move(value)));
        static_assert(hip::std::is_move_constructible<RetT>::value, "");
        static_assert(!hip::std::is_copy_constructible<RetT>::value, "");
        static_assert(!hip::std::is_move_assignable<RetT>::value, "");
        static_assert(!hip::std::is_copy_assignable<RetT>::value, "");
        auto ret = hip::std::not_fn(hip::std::move(value));
        // test it was moved from
        assert(value.value == false);
        // test that ret() negates the original value 'true'
        assert(ret() == false);
        assert(ret(0, 0.0, "blah") == false);
        // Move ret and test that it was moved from and that ret2 got the
        // original value.
        auto ret2 = hip::std::move(ret);
        assert(ret() == true);
        assert(ret2() == false);
        assert(ret2(42) == false);
    }
    {
        using T = CopyCallable<bool>;
        T value(false);
        using RetT = decltype(hip::std::not_fn(value));
        static_assert(hip::std::is_move_constructible<RetT>::value, "");
        static_assert(hip::std::is_copy_constructible<RetT>::value, "");
        static_assert(!hip::std::is_move_assignable<RetT>::value, "");
        static_assert(!hip::std::is_copy_assignable<RetT>::value, "");
        auto ret = hip::std::not_fn(value);
        // test that value is unchanged (copied not moved)
        assert(value.value == false);
        // test 'ret' has the original value
        assert(ret() == true);
        assert(ret(42, 100) == true);
        // move from 'ret' and check that 'ret2' has the original value.
        auto ret2 = hip::std::move(ret);
        assert(ret() == false);
        assert(ret2() == true);
        assert(ret2("abc") == true);
    }
    {
        using T = CopyAssignableWrapper;
        T value(true);
        T value2(false);
        using RetT = decltype(hip::std::not_fn(value));
        static_assert(hip::std::is_move_constructible<RetT>::value, "");
        static_assert(hip::std::is_copy_constructible<RetT>::value, "");
        LIBCPP_STATIC_ASSERT(hip::std::is_move_assignable<RetT>::value, "");
        LIBCPP_STATIC_ASSERT(hip::std::is_copy_assignable<RetT>::value, "");
        auto ret = hip::std::not_fn(value);
        assert(ret() == false);
        auto ret2 = hip::std::not_fn(value2);
        assert(ret2() == true);
#if defined(_LIBCUDACXX_VERSION)
        ret = ret2;
        assert(ret() == true);
        assert(ret2() == true);
#endif // _LIBCUDACXX_VERSION
    }
    {
        using T = MoveAssignableWrapper;
        T value(true);
        T value2(false);
        using RetT = decltype(hip::std::not_fn(hip::std::move(value)));
        static_assert(hip::std::is_move_constructible<RetT>::value, "");
        static_assert(!hip::std::is_copy_constructible<RetT>::value, "");
        LIBCPP_STATIC_ASSERT(hip::std::is_move_assignable<RetT>::value, "");
        static_assert(!hip::std::is_copy_assignable<RetT>::value, "");
        auto ret = hip::std::not_fn(hip::std::move(value));
        assert(ret() == false);
        auto ret2 = hip::std::not_fn(hip::std::move(value2));
        assert(ret2() == true);
#if defined(_LIBCUDACXX_VERSION)
        ret = hip::std::move(ret2);
        assert(ret() == true);
#endif // _LIBCUDACXX_VERSION
    }
}

__host__ __device__
void return_type_tests()
{
    using hip::std::is_same;
    {
        using T = CopyCallable<bool>;
        auto ret = hip::std::not_fn(T{false});
        static_assert(is_same<decltype(ret()), bool>::value, "");
        static_assert(is_same<decltype(ret("abc")), bool>::value, "");
        assert(ret() == true);
    }
    {
        using T = CopyCallable<ExplicitBool>;
        auto ret = hip::std::not_fn(T{true});
        static_assert(is_same<decltype(ret()), bool>::value, "");
        // static_assert(is_same<decltype(ret(hip::std::string("abc"))), bool>::value, "");
        assert(ret() == false);
    }
    {
        using T = CopyCallable<EvilBool>;
        auto ret = hip::std::not_fn(T{false});
        static_assert(is_same<decltype(ret()), EvilBool>::value, "");
        EvilBool_bang_called = 0;
        auto value_ret = ret();
        assert(EvilBool_bang_called == 1);
        assert(value_ret.value == true);
        ret();
        assert(EvilBool_bang_called == 2);
    }
}

// Other tests only test using objects with call operators. Test various
// other callable types here.
__host__ __device__
void other_callable_types_test()
{
    { // test with function pointer
        auto ret = hip::std::not_fn(returns_true);
        assert(ret() == false);
    }
    { // test with lambda
        auto returns_value = [](bool value) { return value; };
        auto ret = hip::std::not_fn(returns_value);
        assert(ret(true) == false);
        assert(ret(false) == true);
    }
    { // test with pointer to member function
        MemFunCallable mt(true);
        const MemFunCallable mf(false);
        auto ret = hip::std::not_fn(&MemFunCallable::return_value);
        assert(ret(mt) == false);
        assert(ret(mf) == true);
        assert(ret(&mt) == false);
        assert(ret(&mf) == true);
    }
    { // test with pointer to member function
        MemFunCallable mt(true);
        MemFunCallable mf(false);
        auto ret = hip::std::not_fn(&MemFunCallable::return_value_nc);
        assert(ret(mt) == false);
        assert(ret(mf) == true);
        assert(ret(&mt) == false);
        assert(ret(&mf) == true);
    }
    { // test with pointer to member data
        MemFunCallable mt(true);
        const MemFunCallable mf(false);
        auto ret = hip::std::not_fn(&MemFunCallable::value);
        assert(ret(mt) == false);
        assert(ret(mf) == true);
        assert(ret(&mt) == false);
        assert(ret(&mf) == true);
    }
}

__host__ __device__
void throws_in_constructor_test()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    struct ThrowsOnCopy {
      ThrowsOnCopy(ThrowsOnCopy const&) {
        throw 42;
      }
      ThrowsOnCopy() = default;
      bool operator()() const {
        assert(false);
#if defined(TEST_COMPILER_C1XX)
        __assume(0);
#else
        __builtin_unreachable();
#endif
      }
    };
    {
        ThrowsOnCopy cp;
        try {
            (void)hip::std::not_fn(cp);
            assert(false);
        } catch (int const& value) {
            assert(value == 42);
        }
    }
#endif
}

__host__ __device__
void call_operator_sfinae_test() {
    { // wrong number of arguments
        using T = decltype(hip::std::not_fn(returns_true));
        static_assert(hip::std::is_invocable<T>::value, ""); // callable only with no args
        static_assert(!hip::std::is_invocable<T, bool>::value, "");
    }
    { // violates const correctness (member function pointer)
        using T = decltype(hip::std::not_fn(&MemFunCallable::return_value_nc));
        static_assert(hip::std::is_invocable<T, MemFunCallable&>::value, "");
        static_assert(!hip::std::is_invocable<T, const MemFunCallable&>::value, "");
    }
    { // violates const correctness (call object)
        using Obj = CopyCallable<bool>;
        using NCT = decltype(hip::std::not_fn(Obj{true}));
        using CT = const NCT;
        static_assert(hip::std::is_invocable<NCT>::value, "");
        static_assert(!hip::std::is_invocable<CT>::value, "");
    }
    // NVRTC appears to be unhappy about... the lambda?
    // but doesn't let me fix it with annotations
#ifndef __CUDACC_RTC__
    { // returns bad type with no operator!
        auto fn = [](auto x) { return x; };
        using T = decltype(hip::std::not_fn(fn));
        static_assert(hip::std::is_invocable<T, bool>::value, "");
        // static_assert(!hip::std::is_invocable<T, hip::std::string>::value, "");
    }
#endif
}

#if 0
__host__ __device__
void call_operator_forwarding_test()
{
    using Fn = ForwardingCallObject;
    auto obj = hip::std::not_fn(Fn{});
    const auto& c_obj = obj;
    { // test zero args
        obj();
        assert(Fn::check_call<>(CT_NonConst | CT_LValue));
        hip::std::move(obj)();
        assert(Fn::check_call<>(CT_NonConst | CT_RValue));
        c_obj();
        assert(Fn::check_call<>(CT_Const | CT_LValue));
        hip::std::move(c_obj)();
        assert(Fn::check_call<>(CT_Const | CT_RValue));
    }
    { // test value categories
        int x = 42;
        const int cx = 42;
        obj(x);
        assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
        obj(cx);
        assert(Fn::check_call<const int&>(CT_NonConst | CT_LValue));
        obj(hip::std::move(x));
        assert(Fn::check_call<int&&>(CT_NonConst | CT_LValue));
        obj(hip::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_NonConst | CT_LValue));
        obj(42);
        assert(Fn::check_call<int&&>(CT_NonConst | CT_LValue));
    }
    { // test value categories - rvalue
        int x = 42;
        const int cx = 42;
        hip::std::move(obj)(x);
        assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
        hip::std::move(obj)(cx);
        assert(Fn::check_call<const int&>(CT_NonConst | CT_RValue));
        hip::std::move(obj)(hip::std::move(x));
        assert(Fn::check_call<int&&>(CT_NonConst | CT_RValue));
        hip::std::move(obj)(hip::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_NonConst | CT_RValue));
        hip::std::move(obj)(42);
        assert(Fn::check_call<int&&>(CT_NonConst | CT_RValue));
    }
    { // test value categories - const call
        int x = 42;
        const int cx = 42;
        c_obj(x);
        assert(Fn::check_call<int&>(CT_Const | CT_LValue));
        c_obj(cx);
        assert(Fn::check_call<const int&>(CT_Const | CT_LValue));
        c_obj(hip::std::move(x));
        assert(Fn::check_call<int&&>(CT_Const | CT_LValue));
        c_obj(hip::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_Const | CT_LValue));
        c_obj(42);
        assert(Fn::check_call<int&&>(CT_Const | CT_LValue));
    }
    { // test value categories - const call rvalue
        int x = 42;
        const int cx = 42;
        hip::std::move(c_obj)(x);
        assert(Fn::check_call<int&>(CT_Const | CT_RValue));
        hip::std::move(c_obj)(cx);
        assert(Fn::check_call<const int&>(CT_Const | CT_RValue));
        hip::std::move(c_obj)(hip::std::move(x));
        assert(Fn::check_call<int&&>(CT_Const | CT_RValue));
        hip::std::move(c_obj)(hip::std::move(cx));
        assert(Fn::check_call<const int&&>(CT_Const | CT_RValue));
        hip::std::move(c_obj)(42);
        assert(Fn::check_call<int&&>(CT_Const | CT_RValue));
    }
    { // test multi arg
        const double y = 3.14;
        // hip::std::string s = "abc";
        // obj(42, hip::std::move(y), s, hip::std::string{"foo"});
        // Fn::check_call<int&&, const double&&, hip::std::string&, hip::std::string&&>(CT_NonConst | CT_LValue);
        // hip::std::move(obj)(42, hip::std::move(y), s, hip::std::string{"foo"});
        // Fn::check_call<int&&, const double&&, hip::std::string&, hip::std::string&&>(CT_NonConst | CT_RValue);
        // c_obj(42, hip::std::move(y), s, hip::std::string{"foo"});
        // Fn::check_call<int&&, const double&&, hip::std::string&, hip::std::string&&>(CT_Const  | CT_LValue);
        // hip::std::move(c_obj)(42, hip::std::move(y), s, hip::std::string{"foo"});
        // Fn::check_call<int&&, const double&&, hip::std::string&, hip::std::string&&>(CT_Const  | CT_RValue);
    }
}
#endif

__host__ __device__
void call_operator_noexcept_test()
{
    {
        using T = ConstCallable<bool>;
        T value(true);
        auto ret = hip::std::not_fn(value);
        static_assert(!noexcept(ret()), "call should not be noexcept");
        auto const& cret = ret;
        static_assert(!noexcept(cret()), "call should not be noexcept");
    }
    {
        using T = NoExceptCallable<bool>;
        T value(true);
        auto ret = hip::std::not_fn(value);
        (void)ret;
        LIBCPP_STATIC_ASSERT(noexcept(!_CUDA_VSTD::__invoke(value)), "");
#if TEST_STD_VER > 14
        static_assert(noexcept(!hip::std::invoke(value)), "");
#endif
// TODO: nvcc gets this wrong, investigate
#if !(defined(__CUDACC__) || defined(__HIPCC__))
        static_assert(noexcept(ret()), "call should be noexcept");
        auto const& cret = ret;
        static_assert(noexcept(cret()), "call should be noexcept");
#endif
    }
    {
        using T = NoExceptCallable<NoExceptEvilBool>;
        T value(true);
        auto ret = hip::std::not_fn(value);
        (void)ret;
// TODO: nvcc gets this wrong, investigate
#if !(defined(__CUDACC__) || defined(__HIPCC__))
        static_assert(noexcept(ret()), "call should not be noexcept");
        auto const& cret = ret;
        static_assert(noexcept(cret()), "call should not be noexcept");
#endif
    }
    {
        using T = NoExceptCallable<EvilBool>;
        T value(true);
        auto ret = hip::std::not_fn(value);
        static_assert(!noexcept(ret()), "call should not be noexcept");
        auto const& cret = ret;
        static_assert(!noexcept(cret()), "call should not be noexcept");
    }
}

__host__ __device__
void test_lwg2767() {
    // See https://cplusplus.github.io/LWG/lwg-defects.html#2767
    struct Abstract { __host__ __device__ virtual void f() const = 0; };
    struct Derived : public Abstract { __host__ __device__ void f() const {} };
    struct F { __host__ __device__ bool operator()(Abstract&&) { return false; } };
    {
        Derived d;
        Abstract &a = d;
        bool b = hip::std::not_fn(F{})(hip::std::move(a));
        assert(b);
    }
}

int main(int, char**)
{
    constructor_tests();
    return_type_tests();
    other_callable_types_test();
    throws_in_constructor_test();
    call_operator_sfinae_test(); // somewhat of an extension
    // call_operator_forwarding_test();
    call_operator_noexcept_test();
    test_lwg2767();

  return 0;
}