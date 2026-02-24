#include "moe_cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define TILE_WIDTH 16
#define MAX_THREADS_PER_BLOCK 512

// A is M × K     B is K × N     →     C is M × N

// A looks like:                  B looks like:
// [                               [
//  row 0: a00 a01 ... a0,k-1       row 0: b00 b01 ... b0,n-1
//  row 1: a10 a11 ... a1,k-1       row 1: b10 b11 ... b1,n-1
//  ...                             ...
//  row m-1                         row k-1
// ]                               ]

// C[row][col] = sum over k ( A[row][k] × B[k][col] )


// Computes C = A @ B  (matrix multiplication, no transpose, alpha=1, beta=0)
// A: M rows × K columns
// B: K rows × N columns
// C: M rows × N columns (output)



// 1. Naive GEMM Kernel (one thread → one output element)
__global__ void sgemm_naive(
    const float* __restrict__ A,   // Input: flattened M × K matrix (row-major)
    const float* __restrict__ B,   // Input: flattened K × N matrix (row-major)
    float* __restrict__ C,         // Output: flattened M × N matrix (row-major)
    int M,                         // # of rows in A and C
    int N,                         // # of columns in B and C
    int K)                         // # of columns in A = # of rows in B (contraction dim)
{
    // Each thread computes exactly one element of the output matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // row index in C (0 to M-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // column index in C (0 to N-1)

    // Skip threads that are outside the valid output region
    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // Inner loop: dot product between row 'row' of A and column 'col' of B
    for (int k = 0; k < K; ++k) {
        // A[row * K + k]  → element A[row, k]
        // B[k * N + col]  → element B[k, col]
        sum += A[row * K + k] * B[k * N + col];
    }

    // Write final result to global memory
    C[row * N + col] = sum;
}



// 2. Tiled (Shared Memory) GEMM – Much faster due to data reuse

#define TILE 32   // Tile size — chosen to match warp size (32 threads) and fit in shared memory

// Computes C = A @ B using shared memory tiling for better memory access & reuse
// Each thread block computes a TILE×TILE sub-tile of C
__global__ void sgemm_tiled(
    const float* __restrict__ A,   // Input: M × K (row-major)
    const float* __restrict__ B,   // Input: K × N (row-major)
    float* __restrict__ C,         // Output: M × N (row-major)
    int M,                         // Rows of A and C
    int N,                         // Columns of B and C
    int K)                         // Columns of A = rows of B
{
    // Shared memory tiles — each holds a TILE×TILE sub-matrix
    __shared__ float sA[TILE][TILE];   // Holds sub-tile of A
    __shared__ float sB[TILE][TILE];   // Holds sub-tile of B (transposed in access pattern)

    int tx = threadIdx.x;   // thread x-index inside block (0..31)
    int ty = threadIdx.y;   // thread y-index inside block (0..31)

    // Global output position this thread block is responsible for
    int row = blockIdx.y * TILE + ty;   // row in C
    int col = blockIdx.x * TILE + tx;   // column in C

    float acc = 0.0f;   // Accumulator for this thread's output element

    // Loop over all tiles along the K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Phase 1: Load data into shared memory (coalesced accesses)
        
        // Load tile of A: rows stay the same, columns advance by tile
        if (row < M && (t * TILE + tx) < K) {
            sA[ty][tx] = A[row * K + (t * TILE + tx)];
        } else {
            sA[ty][tx] = 0.0f;   // Padding for out-of-bounds
        }

        // Load tile of B: note the index swap — we load it "transposed" in shared mem
        // so that later accesses to sB[k][tx] are coalesced for the column direction
        if (col < N && (t * TILE + ty) < K) {
            sB[ty][tx] = B[(t * TILE + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();   // Wait until entire tile is loaded by all threads

        // Phase 2: Compute dot product using the loaded tiles
        for (int k = 0; k < TILE; ++k) {
            // sA[ty][k]  → A elements for this row
            // sB[k][tx]  → B elements for this column (thanks to the way we loaded B)
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();   // Wait before loading next tile (reuse shared mem)
    }

    // Write result only if inside bounds
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}