## Group Members
- Castillo, Marvien Angel C.
- Herrera, Mikhaela Gabrielle B.
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
## i.) Screenshot of the program output with correctness check AND execution time for all cases
## ii.) NSight Screenshots
## iii.) Comparative table of execution time
|2^28 elements <br> CUDA block size = 1024 |	Kernel time (up to the point necessary data to return to error checking part) (do not time the error checking routine) |Speedup vs baseline C program| 
|----------------------------------------- |-------------------------------------------------------------------------------------------------------------------------|-|
| x86-64| | | | 
| X86-64 SIMD XMM | | |
| x86-64 SIMD YMM	| | |
| CUDA Unified | | |
| CUDA Prefetch	| | |
| CUDA Prefetch+page creation	| | | 
| CUDA Prefetch+Page creatition+memadvise	| | |
| CUDA classic MEMCPY	| | |
| CUDA data init in a CUDA kernel	| | |

## iv.) Analysis

### (a.) What overheads are included in the GPU execution time (up to the point where the data is transferred back to the CPU for error checking)? Is it different for each CUDA variant?
-Overheads
These are the additional computation time or resource usage that does not directly contribute to the core computation but is necessary for its execution. The included overheads in the GPU execution time are **Kernel Launch Overhead**, **Memory Access Latency**, **Unified Memory Page Faults** (when using `cudaMallocManaged`), **Prefetching and Memory Advice** (when using `cudaMemPrefetchAsync` and `cudaMemAdvise`), and **Explicit Memory Transfers** (when using `cudaMemcpy`).
-Variant Differences
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
### (c.) Is prefetching always recommended, or should CUDA manage memory?  Give some use cases in which one is better than the other.
### (d.) Between SIMD and SIMT, which one is faster? Give some use cases in which one is better than the other.
## v.) Discuss the problems encountered and solutions made, unique methodology used, AHA moments, etc.
## vi.) Discuss, based on your experience on the particular project use case, the differences between SIMD and SIMT in handling parallelism. <br> Include also the PROS and CONS of using SIMD and SIMT in your use case.
