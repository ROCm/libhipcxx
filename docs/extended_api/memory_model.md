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

# Memory model

Standard C++ presents a view that the cost to synchronize threads is uniform and low.

HIP C++ is different: the cost to synchronize threads grows as threads are further apart.
It is low across threads within a block, but high across arbitrary threads in the system running on multiple GPUs and CPUs.

To account for non-uniform thread synchronization costs that are not always low, HIP C++ extends the standard C++ memory model and concurrency facilities in the `hip::` namespace with **thread scopes**, retaining the syntax and semantics of standard C++ by default.

## Thread Scopes

A _thread scope_ specifies the kind of threads that can synchronize with each other using synchronization primitive such as [`atomic`].

```hip
namespace hip {

enum thread_scope {
  thread_scope_system,
  thread_scope_device,
  thread_scope_block,
  thread_scope_thread
};

}  // namespace hip
```

[`atomic`]: synchronization_primitives/atomic.md

### Scope Relationships

Each program thread is related to each other program thread by one or more thread scope relations:
- Each thread in the system is related to each other thread in the system by the *system* thread scope: `thread_scope_system`.
- Each GPU thread is related to each other GPU thread in the same HIP device by the *device* thread scope: `thread_scope_device`.
- Each GPU thread is related to each other GPU thread in the same HIP thread block by the *block* thread scope: `thread_scope_block`.
- Each thread is related to itself by the `thread` thread scope: `thread_scope_thread`.

## Synchronization primitives

Types in namespaces `std::` and `hip::std::` have the same behavior as corresponding types in namespace `hip::` when instantiated with a scope of `hip::thread_scope_system`.

## Atomicity

An atomic operation is atomic at the scope it specifies if:
- it specifies a scope other than `thread_scope_system`, **or**

the scope is `thread_scope_system` and:

- it affects an object in [managed memory] and [`concurrentManagedAccess`] is `1`, **or**
- it affects an object in CPU memory and [`hostNativeAtomicSupported`] is `1`, **or**
- it is a load or store that affects a naturally-aligned object of sizes `1`, `2`, `4`, or `8` bytes on [mapped memory], **or**
- it affects an object in GPU memory and only GPU threads access it.


Refer to the [HIP programming manual] for more information on [managed memory], [mapped memory], CPU memory, and GPU peer memory.

[mapped memory]: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html#memory-allocation-flags
[managed memory]: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html#managed-memory-allocation
[HIP programming manual]: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/programming_manual.html
[`concurrentManagedAccess`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#abfea758a5672cb7803ac467543c11b67
[`hostNativeAtomicSupported`]: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#a64696b1aa1789ca322e8c86b69b57e7c

## Data Races

Modify [intro.races paragraph 21] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at least one of which is not atomic ***at a scope that includes the thread that performed the other operation***, and neither happens before the other, except for the special case for signal handlers described below. Any such data race results in undefined behavior. [...]

Modify [thread.barrier.class paragraph 4] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> 4. Concurrent invocations of the member functions of `barrier`, other than its destructor, do not introduce data races ***as if they were atomic operations***. [...]

Modify [thread.latch.class paragraph 2] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> 2. Concurrent invocations of the member functions of `latch`, other than its destructor, do not introduce data races ***as if they were atomic operations***.

Modify [thread.sema.cnt paragraph 3] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> 3. Concurrent invocations of the member functions of `counting_semaphore`, other than its destructor, do not introduce data races ***as if they were atomic operations***.

Modify [thread.stoptoken.intro paragraph 5] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> Calls to the functions request_­stop, stop_­requested, and stop_­possible do not introduce data races ***as if they were atomic operations***. [...]

[thread.stoptoken.intro paragraph 5]: https://eel.is/c++draft/thread#stoptoken.intro-5

Modify [atomics.fences paragraph 2 through 4] of ISO/IEC IS 14882 (the C++ Standard) as follows:
> A release fence A synchronizes with an acquire fence B if there exist atomic
> operations X and Y, both operating on some atomic object M, such that A is
> sequenced before X, X modifies M, Y is sequenced before B, and Y reads the
> value written by X or a value written by any side effect in the hypothetical
> release sequence X would head if it were a release operation,
> ***and each operation (A, B, X, and Y) specifies a scope that includes the thread that performed each other operation***.

> A release fence A synchronizes with an atomic operation B that performs an
> acquire operation on an atomic object M if there exists an atomic operation X
> such that A is sequenced before X, X modifies M, and B reads the value
> written by X or a value written by any side effect in the hypothetical
> release sequence X would head if it were a release operation,
> ***and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.

> An atomic operation A that is a release operation on an atomic object M
> synchronizes with an acquire fence B if there exists some atomic operation X
> on M such that X is sequenced before B and reads the value written by A or a
> value written by any side effect in the release sequence headed by A,
> ***and each operation (A, B, and X) specifies a scope that includes the thread that performed each other operation***.

## Example: Message Passing

The following example passes a message stored to the `x` variable by a thread in block `0` to a thread in block `1` via the flag `f`:



<table class="display">
<tr class="header"><td colspan="2" markdown="span" align="center">
<div class="highlight-hip notranslate"><div class="highlight"><pre><span></span>
int x = 0;<br>
int f = 0;
</pre></div>
</div>
</td></tr>
<tr class="header">
<td markdown="span" align="center"> 
<strong>Thread 0 Block 0</strong>
</td><td markdown="span" align="center"> 
<strong>Thread 0 Block 1</strong>
</td>
</tr>
<tr>
<td markdown="span">
<div class="highlight-hip notranslate"><div class="highlight"><pre><span></span>
x = 42;<br>
hip::atomic_ref<int, hip::thread_scope_device> flag(f);<br>
flag.store(1, memory_order_release);
</pre></div>
</div>
</td>
</td>
<td markdown="span">
<div class="highlight-hip notranslate"><div class="highlight"><pre><span></span>
hip::atomic_ref<int, hip::thread_scope_device> flag(f);<br>
while(flag.load(memory_order_acquire) != 1);<br>
assert(x == 42);
</pre></div>
</div>
</td>
</tr>
</table>

In the following variation of the previous example, two threads concurrently access the `f` object without synchronization, which leads to a **data race**, and exhibits **undefined behavior**:

<table>
<tr><td colspan="2" markdown="span" align="center">
<div class="highlight-hip notranslate"><div class="highlight"><pre><span></span>
int x = 0;<br>
int f = 0;
</pre></div>
</div>
</td></tr>
<tr>
<td markdown="span" align="center"> 
<strong>Thread 0 Block 0</strong>
</td><td markdown="span" align="center"> 
<strong>Thread 0 Block 1</strong>
</td>
</tr>
<tr>
<td markdown="span">
<div class="highlight-hip notranslate"><div class="highlight"><pre><span></span>
x = 42;<br>
hip::atomic_ref<int, hip::thread_scope_block> flag(f);<br>
flag.store(1, memory_order_release);  // UB: data race
</pre></div>
</div>
</td>
<td markdown="span">
<div class="highlight-hip notranslate"><div class="highlight"><pre><span></span>
hip::atomic_ref<int, hip::thread_scope_device> flag(f);<br>
while(flag.load(memory_order_acquire) != 1); // UB: data race<br>
assert(x == 42);
</pre></div>
</div>
</td>
</tr>
</table>

While the memory operations on `f` - the store and the loads - are atomic, the scope of the store operation is "block scope". Since the store is performed by Thread 0 of Block 0, it only includes all other threads of Block 0. However, the thread doing the loads is in Block 1, i.e., it is not in a scope included by the store operation performed in Block 0, causing the store and the load to not be "atomic", and introducing a data-race. 

[intro.races paragraph 21]: https://eel.is/c++draft/intro.races#21
[thread.barrier.class paragraph 4]: https://eel.is/c++draft/thread.barrier.class#4
[thread.latch.class paragraph 2]: https://eel.is/c++draft/thread.latch.class#2
[thread.sema.cnt paragraph 3]: https://eel.is/c++draft/thread.sema.cnt#3
[atomics.fences paragraph 2 through 4]: https://eel.is/c++draft/atomics.fences#2
