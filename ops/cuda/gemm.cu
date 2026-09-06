// Small FP32 GEMM kernels for the controlled CUDA comparison.
// Each block computes a 16x16 tile and uses shared memory for both operands.
#define TILE 16

extern "C" __global__ void gemm_nt(
    const float* a, const float* b_rows, float* c,
    int M, int N, int K) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < K; base += TILE) {
        int ak = base + threadIdx.x;
        int bk = base + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? a[row * K + ak] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (col < N && bk < K) ? b_rows[col * K + bk] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) c[row * N + col] = sum;
}

extern "C" __global__ void gemm_nn(
    const float* a, const float* b, float* c,
    int M, int N, int K) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < N; base += TILE) {
        int ak = base + threadIdx.x;
        int bk = base + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = (row < M && ak < N) ? a[row * N + ak] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (bk < N && col < K) ? b[bk * K + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < K) c[row * K + col] = sum;
}

extern "C" __global__ void gemm_tn(
    const float* a, const float* b, float* c,
    int M, int N, int K) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int base = 0; base < M; base += TILE) {
        int ak = base + threadIdx.x;
        int bk = base + threadIdx.y;
        tile_a[threadIdx.y][threadIdx.x] = (ak < M && row < N) ? a[ak * N + row] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (bk < M && col < K) ? b[bk * K + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < K) c[row * K + col] = sum;
}
