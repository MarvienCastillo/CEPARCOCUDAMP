## 0.) Group Members
- Castillo, Marvien Angel C.
- Herrera, Mikhaela Gabrielle B.
- Regindin, Sean Adrien I.
## Project Specification
SIMD Project Specifications
Write the kernel in (1) C program; (2) an x86-64 assembly language; (3) x86 SIMD AVX2 assembly language using XMM register; (4) x86 SIMD AVX2 assembly language using YMM register. The kernel is to perform **element-wise maximum with index tracking**.

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
