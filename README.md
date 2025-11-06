## 0.) Group Members
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

## iv.) Analysis of results
## v.) Discuss the problems encountered and solutions made, unique methodology used, AHA moments, etc.
## vi.) Discuss, based on your experience on the particular project use case, the differences between SIMD and SIMT in handling parallelism. <br> Include also the PROS and CONS of using SIMD and SIMT in your use case.
