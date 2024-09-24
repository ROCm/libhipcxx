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


# Standard API

## Synchronization Library

Any Standard C++ header not listed below is omitted/not supported.

| API    | Description |
| -------------------------------------------------------------------------- | ---------------------------------- |
| [`<hip/std/atomic>`]    | Atomic objects and operations (see also: [Extended API](./extended_api/synchronization_primitives/atomic.md)). |

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
| [`<hip/std/cassert>`] | Lightweight assumption testing.          |
| [`<hip/std/cstddef>`] | Fundamental types.  |


[`<hip/std/atomic>`]: https://en.cppreference.com/w/cpp/header/atomic
[`<hip/std/latch>`]: https://en.cppreference.com/w/cpp/header/latch
[`<hip/std/barrier>`]: https://en.cppreference.com/w/cpp/header/barrier
[`<hip/std/semaphore>`]: https://en.cppreference.com/w/cpp/header/semaphore
[`<hip/std/cassert>`]: https://en.cppreference.com/w/cpp/header/cassert
[`<hip/std/cstddef>`]: https://en.cppreference.com/w/cpp/header/cstddef

