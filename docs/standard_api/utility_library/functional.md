<!-- MIT License
  -- 
  -- Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
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


# `<hip/std/functional>`

## Omissions

The following facilities in section [functional.syn] of ISO/IEC IS 14882 (the
  C++ Standard) are not available in libhipcxx today:

- [`std::function`] - Polymorphic function object wrapper.
- [`std::bind`] - Generic function object binder / lambda facility.
- [`std::reference_wrapper`] - Reference wrapper type.
- [`std::hash`] - Hash function object.

### `std::function`

[`std::function`] is a polymorphic function object wrapper.
Implementing it requires both polymorphism (either hand built dispatch tables
  or the use of C++ virtual functions) and memory allocation.
This means that it is non-trivial to implement a heterogeneous version of this
  facility today.
As such, we have deferred it.

### `std::bind`

[`std::bind`] is a general-purpose function object binder / lambda facility.
It relies on constexpr global variables for placeholders, which presents
  heterogeneous implementation challenges.
Therefore, we've decided to hold off on providing this feature for now.

### `std::hash`

[`std::hash`] is a function object which hashes entities.
While this is an important feature, it is also important that we pick a hash
  implementation that makes sense for GPUs.
That implementation might be different from the default that the upstream
  libc++ uses.
Further research and investigation is required before we can provide this
  feature.

### `std::reference_wrapper`

[`std::reference_wrapper`] is a [*CopyConstructible*] and
  [*CopyAssignable*] wrapper around a reference to an object or function of
  type `T`.
There is nothing that makes this facility difficult to implement heterogeneously
  today.
It is a value type that does not allocate memory, hold
  pointers, have virtual functions, or make calls to platform specific APIs.

No design or functional changes were required to port the upstream libc++
  implementations of this facility.
We just had to add execution space specifiers to port it.

However, this feature failed tests involving function pointers with some of the
  compilers we support.
So, we've omitted this feature for now.


[functional.syn]: https://eel.is/c++draft/functional.syn

[*CopyConstructible*]: https://eel.is/c++draft/utility.arg.requirements#:requirements,Cpp17CopyConstructible
[*CopyAssignable*]: https://eel.is/c++draft/utility.arg.requirements#:requirements,Cpp17CopyAssignable

[`std::function`]: https://en.cppreference.com/w/cpp/utility/functional/function
[`std::bind`]: https://en.cppreference.com/w/cpp/utility/functional/bind
[`std::reference_wrapper`]: https://en.cppreference.com/w/cpp/utility/functional/reference_wrapper
[`std::hash`]: https://en.cppreference.com/w/cpp/utility/hash

