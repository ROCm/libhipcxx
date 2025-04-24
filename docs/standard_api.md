<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
  -- 
  -- Permission is hereby granted, free of charge, to any person obtaining a copy
  -- of this software and associated documentation files (the "Software"), to deal
  -- in the Software without restriction, including without limitation the rights
  -- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  -- copies of the Software, and to permit persons to whom the Software is
  -- furnished to do so, subject to the following conditions:
  -- 
  -- The above copyright notice and this permission notice shall be included in all
  -- copies or substantial portions of the Software.
  -- 
  -- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  -- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  -- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  -- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  -- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  -- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  -- SOFTWARE.
  -->


# Standard API

## Standard Library Backports

C++ Standard versions include new language features and new library features.
As the name implies, language features are new features of the language the require compiler support.
Library features are simply new additions to the Standard Library that typically do not rely on new language features nor require compiler support and could conceivably be implemented in an older C++ Standard.
Typically, library features are only available in the particular C++ Standard version (or newer) in which they were introduced, even if the library features do not depend on any particular language features.

In effort to make library features available to a broader set of users, the NVIDIA C++ Standard Library relaxes this restriction.
libcu++ makes a best-effort to provide access to C++ Standard Library features in older C++ Standard versions than they were introduced.
For example, the calendar functionality added to `<chrono>` in C++20 is made available in C++14.

Feature availability:
- C++17 and C++20 features of`<chrono>` available in C++14:
  -  calendar functionality, e.g., `year`,`month`,`day`,`year_month_day`
  -  duration functions, e.g., `floor`, `ceil`, `round`
  -  Note: Timezone and clocks added in C++20 are not available
- C++17 features from `<type_traits>` available in C++14:
  - Convenience `_v` aliases such as `is_same_v`
  - `void_t`
  - Trait operations: `conjunction`,`negation`,`disjunction`
  - `invoke_result`
- C++20 constexpr `<complex>` is available in C++14.
  - all operation on complex are made constexpr if `is_constant_evaluated` is supported.
- C++20 `<concepts>` are available in C++14.
  - all standard concepts are available in C++14 and C++17. However, they need to be used similar to type traits as language concepts are not available.
- C++20 `<span>` is mostly available in C++14.
  - With the exception of the range based constructors all features are available in C++14 and C++17. The range based constructors are emulated but not 100% equivalent.
- C++20 features of `<functional>` have been partially ported to C++17.
  - `bind_front` is available in C++17.
- C++23 `<mdspan>` is available in C++17.
  - mdspan is feature complete in C++17 onwards.
  - mdspan on msvc is only supported in C++20 and onwards.

## Synchronization Library

Any Standard C++ header not listed below is omitted/not supported.

| API    | Description |
| -------------------------------------------------------------------------- | ---------------------------------- |
| [`<cuda/std/atomic>`]    | Atomic objects and operations (see also: [Extended API](./extended_api/synchronization_primitives/atomic.md)). |

## Numerics Library

See [Numerics Library](./standard_api/numerics_library.md)

## Utility Library

See [Utility Library](./standard_api/utility_library.md)

## Time Library

See [Time Library](./standard_api/time_library.md)

## C Library

Any Standard C++ header not listed below is omitted.

| API    | Description |
| -------------------------------------------------------------------------- | ---------------------------------- |
| [`<cuda/std/cassert>`] | Lightweight assumption testing.          |
| [`<cuda/std/cstddef>`] | Fundamental types.  |


[`<cuda/std/atomic>`]: https://en.cppreference.com/w/cpp/header/atomic
[`<cuda/std/latch>`]: https://en.cppreference.com/w/cpp/header/latch
[`<cuda/std/barrier>`]: https://en.cppreference.com/w/cpp/header/barrier
[`<cuda/std/semaphore>`]: https://en.cppreference.com/w/cpp/header/semaphore
[`<cuda/std/cassert>`]: https://en.cppreference.com/w/cpp/header/cassert
[`<cuda/std/cstddef>`]: https://en.cppreference.com/w/cpp/header/cstddef
