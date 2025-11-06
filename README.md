## Group Members
- Castillo, Marvien Angel C.
- Herrera, Mikaela Gabrielle B.
- Regindin, Sean Adrien I.
## Project Specification
- This project explores the performance and correctness of various CUDA implementations based on our SIMD specification. We compare multiple CUDA variants against a baseline C/C++ implementation, analyzing execution time, correctness, and architectural behavior.
- Implementations includes:
  - (1) a C/C++ program version
  - (2) a CUDA program version using a grid-stride loop without prefetch and without mem advise
  - (3) a CUDA program version using a grid-stride loop with prefetching but without page creation and without mem advise
  - (4) a CUDA program version using a grid-stride loop with prefetch, with page creation but without mem advise
  - (5) a CUDA program version using a grid-stride loop with prefetch, with page creation and with mem advise
  - (6) Classic MemCopy method (no Unified memory)
  - (7) Another CUDA kernel that initializes the data  


**Input**: Scalar variable n (integer) contains the length of the vector; Vectors A and B are **single-precision float**.

Process:
For each index i, compute:
```
c[i] = max(A[i], B[i])

idx[i] = 0 if A[i] >= B[i], else 1
```

**Output**: store maximum of each element in single-precision vector C. Store index in 32-bit vector idx.
## AI Usage Declaration
- AI is applied to content rewriting or paraphrasing.
## i.) Screenshot of the program output with correctness check AND execution time for all cases
### x86-64
<img width="1280" height="677" alt="image" src="https://github.com/user-attachments/assets/7faff59c-5e96-4db7-b3b4-e1a6cf32e1a0" />

### x86-64 SIMD XMM
<img width="1280" height="678" alt="image" src="https://github.com/user-attachments/assets/df366f46-7f9e-404a-be66-98b529fc2b4c" />

### x86-64 SIMD YMM
### VARIANT 1: C
<img width="1131" height="543" alt="image" src="https://github.com/user-attachments/assets/d0eaaf4b-d3f9-46c1-b67d-3d1029af38d4" />

### VARIANT 2: CUDA Grid-stride Loop
<img width="1178" height="701" alt="image" src="https://github.com/user-attachments/assets/100eeb18-e87f-4fa6-8813-cc935947bf91" />

### VARIANT 3: CUDA Prefetch
<img width="1173" height="763" alt="image" src="https://github.com/user-attachments/assets/21686bc7-3999-427a-bb04-3114fac42281" />

### VARIANT 4: CUDA Prefetch + page creation
<img width="1131" height="740" alt="image" src="https://github.com/user-attachments/assets/960573fa-af84-4eb8-a682-28f1ade3e4b8" />

### VARIANT 5: CUDA Prefetch + Page creation + memadvise
<img width="1161" height="748" alt="image" src="https://github.com/user-attachments/assets/ee2b7d61-90c5-4991-915f-fd51f864a020" />

### VARIANT 6: CUDA classic MEMCPY
<img width="1212" height="656" alt="image" src="https://github.com/user-attachments/assets/74219c83-f3fe-42cf-8726-b5b6605dc466" />

### VARIANT 7: CUDA data init in a CUDA kernel
<img width="1143" height="707" alt="image" src="https://github.com/user-attachments/assets/452e6737-9dd0-4d00-8613-0760d7f2e81c" />


## ii.) NSight Screenshots

### VARIANT 2: CUDA Unified
<img width="1280" height="679" alt="image" src="https://github.com/user-attachments/assets/7eb6171f-37e8-4c18-95c3-24d5ec43d4b5" />

### VARIANT 3: CUDA Prefetch
<img width="1919" height="1020" alt="image" src="https://github.com/user-attachments/assets/aff5513a-c475-4789-9c5b-d78f8c17510d" />

### VARIANT 4: CUDA Prefetch + page creation
<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/8d351b61-d01a-4909-a03a-c36ceb9cee70" />

### VARIANT 5: CUDA Prefetch + Page creation + memadvise
<img width="1919" height="1017" alt="image" src="https://github.com/user-attachments/assets/298ac6a3-4585-4e11-a01a-bdd67ffdf897" />

### VARIANT 6: CUDA classic MEMCPY
<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/8d42ab5d-452c-4c2f-b203-5ebec6af9212" />

### VARIANT 7: CUDA data init in a CUDA kernel
<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/533c33f8-c15a-445b-9db0-54423c0c720e" />

## iii.) Comparative table of execution time
#### Baseline C execution time: 4648.745000 ms
#### Speedup = Baseline C time / Kernel time

|2^28 elements <br> CUDA block size = 1024 |	Kernel time (up to the point necessary data to return to error checking part) (do not time the error checking routine) |Speedup vs baseline C program| 
|----------------------------------------- |-------------------------------------------------------------------------------------------------------------------------|-|
| x86-64| 1172.539883ms | 3.96 | 
| X86-64 SIMD XMM | 986.44755ms | 4.71 |
| x86-64 SIMD YMM	| a |  |
| CUDA Grid-stride Loop | 278.45ms | 16.70 |
| CUDA Prefetch	| 14.204ms | 327.28 |
| CUDA Prefetch + page creation	| 14.375ms | 323.39 | 
| CUDA Prefetch + Page creation + memadvise	| 13.846ms | 335.75 |
| CUDA classic MEMCPY	| 11.257ms |412.89 |
| CUDA data init in a CUDA kernel	| 120.76ms | 38.50 |

## iv.) Analysis

### (a.) What overheads are included in the GPU execution time (up to the point where the data is transferred back to the CPU for error checking)? Is it different for each CUDA variant?
-Overheads are the additional computation time or resource usage that does not directly contribute to the core computation but is necessary for its execution. The included overheads in the GPU execution time are **Kernel Launch Overhead**, **Memory Access Latency**, **Unified Memory Page Faults** (when using `cudaMallocManaged`), **Prefetching and Memory Advice** (when using `cudaMemPrefetchAsync` and `cudaMemAdvise`), and **Explicit Memory Transfers** (when using `cudaMemcpy`).
The project includes the following implementations to explore performance and memory management strategies:
1. **Baseline C/C++ Program**
  - Performs element-wise maximum with index tracking on the CPU.
  - serves as a reference point, executing the element-wise maximum and index tracking on the CPU. While straightforward, it lacks the parallelism and throughput of GPU-based solutions.
2. **CUDA Variant 1: Grid-Stride Loop (No Prefetch, No Memory Advice)**
- Uses cudaMallocManaged without any memory optimization.
- uses a grid-stride loop with unified memory but without prefetching or memory advice
- is simple to implement but suffers from runtime page faults, leading to unpredictable performance.
3. **CUDA Variant 2: Grid-Stride Loop + Prefetching (No Page Creation, No Memory Advice)**
- Adds cudaMemPrefetchAsync to reduce runtime page faults.
- reduces page faults by migrating data before kernel execution. This improves consistency and lowers latency, especially for large datasets.
4. **CUDA Variant 3: Grid-Stride Loop + Prefetching + Page Creation (No Memory Advice)**
- Incorporates page creation to improve memory locality.
- pre-allocates memory pages on the device. This further reduces migration overhead and improves memory locality.
5. **CUDA Variant 4: Grid-Stride Loop + Prefetching + Page Creation + Memory Advice**
- Uses cudaMemAdvise to guide memory placement and access behavior.
- guiding the system on access patterns and preferred memory placement. This is particularly effective on systems with complex memory hierarchies, delivering the best performance among unified memory variants.
6. **CUDA Variant 5: Classic Memory Copy (No Unified Memory)**
- Allocates separate host/device memory and uses cudaMemcpy for transfers.
- allocating separate host and device memory and transferring data explicitly.
- Although this approach involves more detailed coding, it helps avoid hidden performance issues and gives precise control over how memory is managed.
7. **CUDA Variant 6: Device-Side Data Initialization Kernel**
- eliminating host-to-device transfers. This is efficient for workloads where data generation is part of the computation, and it keeps all memory operations within the GPU.

### (b.) How does block size affect execution time (observing various elements and using max blocks)?  Which block size will you recommend?
Block size plays a crucial role in shaping how threads are distributed, how memory is accessed, and how effectively the GPU's resources are used. Based on our benchmarking experiments, we found that execution time changes noticeably with different block sizes, especially when working with large datasets such as vectors containing millions of elements. In CUDA, threads are organized into blocks, which are then assigned to streaming multiprocessors. The number of threads per block influences several key performance factors. Larger block sizes tend to improve occupancy—the proportion of active warps relative to the hardware’s maximum capacity—resulting in better utilization of GPU cores. They also enhance memory coalescing when threads access adjacent memory locations, which helps reduce memory latency. 

However, if the block size is too large, it can exceed hardware limits for registers or shared memory. This may lead to register spilling into slower global memory or reduced occupancy due to resource constraints.
Importantly, there is no one-size-fits-all block size. The optimal configuration depends on the kernel’s computational complexity, memory demands, and the specific architecture of the GPU being used.
### (c.) Is prefetching always recommended, or should CUDA manage memory?  Give some use cases in which one is better than the other.
While prefetching can significantly improve performance in many scenarios, it is not always the best choice. Prefetching introduces a small setup overhead, as it requires issuing `cudaMemPrefetchAsync` calls and potentially synchronizing streams. In return, it reduces or eliminates runtime page faults by proactively migrating data to the device before kernel execution. This leads to more predictable performance and lower latency, especially for large arrays or compute-intensive kernels.

However, in some cases, the default behavior of CUDA’s Unified Memory manager is sufficient—or even preferable. For example, if the dataset is small or if the kernel accesses only a subset of the data, the overhead of prefetching may outweigh its benefits.
### (d.) Between SIMD and SIMT, which one is faster? Give some use cases in which one is better than the other.
In general, SIMT is faster for large-scale, massively parallel workloads. SIMT, as implemented in CUDA, allows thousands of threads to execute concurrently across multiple streaming multiprocessors. Each thread operates independently, enabling flexible control flow and efficient scaling across large datasets. This makes SIMT ideal for GPU-based tasks such as image processing, scientific simulations, and deep learning.

SIMD, on the other hand, is limited by the width of the vector registers (e.g., 128-bit XMM or 256-bit YMM in AVX2). It executes the same instruction across multiple data elements in lockstep, which is efficient for small, tightly packed data but less scalable for large arrays. SIMD is typically faster for short, latency-sensitive operations on the CPU, especially when the data fits within cache and branching is minimal.

In cases where the needed requirement is a large-scale data, image or graphics processing, or machine learning, it is better to use SIMT because it solves most of the problem faster than SIMD such as SIMT handles millions of elements efficiently, making it ideal for tasks like element-wise vector operations, sorting, and reduction. Also, SIMT’s thread-level parallelism maps naturally to pixel-level computations. In addition, SIMT supports complex control flow and high throughput, essential for training and inference workloads.

On the other hand, SIMD could be the better choice than SIMT when there is a need for Real-time signal processing, Small matrix/vector operations, Embedded systems and mobile CPUs. SIMD excels in low-latency environments where operations must complete within strict timing constraints. Also, Tasks like dot products or small convolution kernels benefit from SIMD’s low overhead and tight memory locality. Lastly, SIMD is often the only available parallelism model in constrained environments.

## v.) Discuss the problems encountered and solutions made, unique methodology used, AHA moments, etc.
The problems we encountered was the amount of elements that is needed for the table. For the YMM, it took so much time so we decided to just loop it once instead of looping it 30. The problems with YMM stems from our previous SIMD project wherein we implemented it incorrectly therefore it took so much time when it should be faster than XMM. Furthermore, since the memCPY was not discussed in the class, we did a lot of research in order to get the solution for it. 
## vi.) Discuss, based on your experience on the particular project use case, the differences between SIMD and SIMT in handling parallelism. <br> Include also the PROS and CONS of using SIMD and SIMT in your use case.
Based on the table above, we have seen that SIMT is so much faster than SIMD. This is because SIMT involves CPU and GPU while SIMD only involves CPU. However there are still pros and cons between the two.

### SIMD Pros: 
- SIMD is much easier to program because it only uses AVX2 instructions which mostly use a less amount of lines.  
### SIMD Cons:
- SIMD is slower when the dataset is high. Which is also why the execution for YMM and XMM in this table is much slower than CUDA since it uses 2^28 number of elements
### SIMT Pros:
- SIMT excels with large datasets because the CPU and GPU cooperate, with the GPU providing the massive parallel throughput that the CPU orchestrates.
### SIMT Cons:
- SIMT is harder to program because it uses CUDA and memory allocation.

