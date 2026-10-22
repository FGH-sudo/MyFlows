// MyFlows native CUDA scheduler.
//
// CuPy owns every device allocation. This DLL compiles the project's kernels with NVRTC once per
// process for the current device's architecture (caching the binary on disk) and launches them with
// the CUDA driver API on the caller's stream. GEMMs run on cuBLAS or on the register-tiled kernels
// below, chosen per call. It links against nothing but the Windows loader: the driver, NVRTC and
// cuBLAS are resolved at run time.
//
// Every failure records "line: call: status name (description)" in a thread-local message that
// Python reads through mf_last_error(). MYFLOWS_NATIVE_SYNC=1 (or mf_set_debug_sync) synchronizes
// the stream after each launch so asynchronous faults are reported by the kernel that caused them.
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = std::uint64_t;
using nvrtcResult = int;
using nvrtcProgram = void*;
using cublasHandle_t = void*;
using cublasStatus_t = int;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr nvrtcResult NVRTC_SUCCESS = 0;
constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;
constexpr int CUBLAS_DEFAULT_MATH = 0;  // IEEE FP32 GEMM; TF32 tensor-core math stays disabled.
constexpr int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
constexpr std::int64_t INDEX_LIMIT = 2147483647;  // kernels index every buffer with int32
constexpr unsigned GRID_YZ_LIMIT = 65535;

enum Status : int {
    MF_OK = 0,
    MF_ERR_INIT = 1,
    MF_ERR_ARG = 2,
    MF_ERR_CUDA = 3,
    MF_ERR_CUBLAS = 4,
    MF_ERR_CONTEXT = 5,
};

// Activation codes shared with the kernels and native.py.
enum Activation : int { ACT_NONE = 0, ACT_RELU = 1, ACT_LEAKY_RELU = 2 };

using CuModuleLoadData = CUresult (*)(CUmodule*, const void*);
using CuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuLaunchKernel = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned, unsigned,
                                    CUstream, void**, void**);
using CuStreamSynchronize = CUresult (*)(CUstream);
using CuCtxGetCurrent = CUresult (*)(CUcontext*);
using CuCtxGetDevice = CUresult (*)(CUdevice*);
using CuDeviceGetAttribute = CUresult (*)(int*, int, CUdevice);
using CuGetErrorText = CUresult (*)(CUresult, const char**);
using NvrtcCreateProgram = nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int,
                                           const char* const*, const char* const*);
using NvrtcCompileProgram = nvrtcResult (*)(nvrtcProgram, int, const char* const*);
using NvrtcGetSize = nvrtcResult (*)(nvrtcProgram, std::size_t*);
using NvrtcGetBytes = nvrtcResult (*)(nvrtcProgram, char*);
using NvrtcDestroyProgram = nvrtcResult (*)(nvrtcProgram*);
using NvrtcVersion = nvrtcResult (*)(int*, int*);
using NvrtcGetNumSupportedArchs = nvrtcResult (*)(int*);
using NvrtcGetSupportedArchs = nvrtcResult (*)(int*);
using NvrtcGetErrorString = const char* (*)(nvrtcResult);
using CublasCreate = cublasStatus_t (*)(cublasHandle_t*);
using CublasSetStream = cublasStatus_t (*)(cublasHandle_t, CUstream);
using CublasSetMathMode = cublasStatus_t (*)(cublasHandle_t, int);
using CublasSgemm = cublasStatus_t (*)(cublasHandle_t, int, int, int, int, int,
                                       const float*, const float*, int,
                                       const float*, int, const float*, float*, int);
using CublasGetStatusText = const char* (*)(cublasStatus_t);

struct Api {
    CuModuleLoadData cuModuleLoadData = nullptr;
    CuModuleGetFunction cuModuleGetFunction = nullptr;
    CuLaunchKernel cuLaunchKernel = nullptr;
    CuStreamSynchronize cuStreamSynchronize = nullptr;
    CuCtxGetCurrent cuCtxGetCurrent = nullptr;
    CuCtxGetDevice cuCtxGetDevice = nullptr;
    CuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
    CuGetErrorText cuGetErrorName = nullptr;    // optional
    CuGetErrorText cuGetErrorString = nullptr;  // optional
    NvrtcCreateProgram nvrtcCreateProgram = nullptr;
    NvrtcCompileProgram nvrtcCompileProgram = nullptr;
    NvrtcGetSize nvrtcGetProgramLogSize = nullptr;
    NvrtcGetBytes nvrtcGetProgramLog = nullptr;
    NvrtcGetSize nvrtcGetPTXSize = nullptr;
    NvrtcGetBytes nvrtcGetPTX = nullptr;
    NvrtcGetSize nvrtcGetCUBINSize = nullptr;  // optional
    NvrtcGetBytes nvrtcGetCUBIN = nullptr;     // optional
    NvrtcDestroyProgram nvrtcDestroyProgram = nullptr;
    NvrtcVersion nvrtcVersion = nullptr;
    NvrtcGetNumSupportedArchs nvrtcGetNumSupportedArchs = nullptr;  // optional
    NvrtcGetSupportedArchs nvrtcGetSupportedArchs = nullptr;        // optional
    NvrtcGetErrorString nvrtcGetErrorString = nullptr;              // optional
    CublasCreate cublasCreate = nullptr;
    CublasSetStream cublasSetStream = nullptr;
    CublasSetMathMode cublasSetMathMode = nullptr;
    CublasSgemm cublasSgemm = nullptr;
    CublasGetStatusText cublasGetStatusName = nullptr;  // optional
};

struct Kernel {
    const char* name;
    CUfunction function;
};

Api api;
HMODULE driver_module = nullptr;
HMODULE nvrtc_module = nullptr;
HMODULE cublas_module = nullptr;
CUmodule cuda_module = nullptr;
CUcontext init_context = nullptr;
cublasHandle_t cublas_handle = nullptr;
std::mutex init_mutex;
bool initialized = false;
bool debug_sync = false;
int sm_count = 0;
std::string info_text = "{}";
thread_local std::string last_error;

// ---------------------------------------------------------------------------------------------
// Kernels. Every buffer is indexed with int32; the host side rejects larger problems first.
// ---------------------------------------------------------------------------------------------
const char* kernel_source = R"CUDA(
#define MF_TILE 32
#define MF_ROWS 8

extern "C" __global__ void im2col_forward(
    const float* x, float* cols,
    int N, int CI, int H, int W, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int features = CI * KH * KW;
    int rows = N * OH * OW;
    if (idx >= rows * features) return;
    int feature = idx % features;
    int row = idx / features;
    int kw = feature % KW;
    int kh = (feature / KW) % KH;
    int ic = feature / (KH * KW);
    int ow = row % OW;
    int oh = (row / OW) % OH;
    int n = row / (OH * OW);
    int ih = oh * SH - PH + kh;
    int iw = ow * SW - PW + kw;
    cols[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
        ? x[((n * CI + ic) * H + ih) * W + iw]
        : 0.0f;
}

extern "C" __global__ void col2im_backward(
    const float* grad_cols, float* dx,
    int N, int CI, int H, int W, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * CI * H * W) return;
    int iw = idx % W;
    int ih = (idx / W) % H;
    int ic = (idx / (W * H)) % CI;
    int n = idx / (W * H * CI);
    float sum = 0.0f;
    int oh0 = max(0, (ih + PH - KH + SH) / SH);
    int oh1 = min(OH - 1, (ih + PH) / SH);
    int ow0 = max(0, (iw + PW - KW + SW) / SW);
    int ow1 = min(OW - 1, (iw + PW) / SW);
    for (int oh = oh0; oh <= oh1; ++oh)
        for (int ow = ow0; ow <= ow1; ++ow) {
            int kh = ih + PH - oh * SH;
            int kw = iw + PW - ow * SW;
            int row = (n * OH + oh) * OW + ow;
            int feature = (ic * KH + kh) * KW + kw;
            sum += grad_cols[row * (CI * KH * KW) + feature];
        }
    dx[idx] = sum;
}

// GEMM rows are NHWC (N*OHW, CO). One pass writes the NCHW output with the bias and the fused
// activation; the transpose goes through a padded shared-memory tile so that both the read
// (along CO) and the write (along OHW) are coalesced and the tile has no bank conflicts.
// LeakyReLU is only fused for alpha > 0, so y > 0 exactly when the pre-activation is > 0.
extern "C" __global__ void conv_epilogue_nchw(
    const float* rows, const float* bias, float* y, int OHW, int CO, int act, float alpha) {
    __shared__ float tile[MF_TILE][MF_TILE + 1];
    const int n = blockIdx.z, p0 = blockIdx.x * MF_TILE, c0 = blockIdx.y * MF_TILE;
    const float* src = rows + (long long)n * OHW * CO;
    float* dst = y + (long long)n * CO * OHW;
    for (int j = threadIdx.y; j < MF_TILE; j += MF_ROWS) {
        const int p = p0 + j, c = c0 + threadIdx.x;
        if (p < OHW && c < CO) tile[threadIdx.x][j] = src[(long long)p * CO + c];
    }
    __syncthreads();
    for (int j = threadIdx.y; j < MF_TILE; j += MF_ROWS) {
        const int c = c0 + j, p = p0 + threadIdx.x;
        if (p < OHW && c < CO) {
            float v = tile[j][threadIdx.x];
            if (bias) v += bias[c];
            if (act == 1) v = v < 0.f ? 0.f : v;  // keeps NaN, like numpy.maximum(0, v)
            else if (act == 2) v = v > 0.f ? v : alpha * v;
            dst[(long long)c * OHW + p] = v;
        }
    }
}

// Backward counterpart: dy (N, CO, OHW) -> dyT (N*OHW, CO) for the GEMMs, gated by the fused
// activation through the forward output y, while every block accumulates its per-channel sums
// of the gated dy in FP64. A block walks the spatial tiles blockIdx.x, blockIdx.x + gridDim.x,
// ... in a fixed order and reduces its 32 lanes with a fixed shuffle tree, so the partial sums
// partial[(n * gridDim.x + blockIdx.x) * CO + c] (and hence db) are bitwise reproducible.
extern "C" __global__ void conv_grad_rows(
    const float* dy, const float* y, float* dyT, double* partial,
    int OHW, int CO, int act, float alpha) {
    __shared__ float tile[MF_TILE][MF_TILE + 1];
    const int n = blockIdx.z, c0 = blockIdx.y * MF_TILE;
    const float* src = dy + (long long)n * CO * OHW;
    const float* gate = y ? y + (long long)n * CO * OHW : nullptr;
    float* dst = dyT + (long long)n * OHW * CO;
    double acc[MF_TILE / MF_ROWS] = {0.0, 0.0, 0.0, 0.0};
    for (int p0 = blockIdx.x * MF_TILE; p0 < OHW; p0 += gridDim.x * MF_TILE) {
        int k = 0;
        for (int j = threadIdx.y; j < MF_TILE; j += MF_ROWS, ++k) {
            const int c = c0 + j, p = p0 + threadIdx.x;
            float v = 0.f;
            if (p < OHW && c < CO) {
                const long long at = (long long)c * OHW + p;
                v = src[at];
                if (act == 1) v = gate[at] > 0.f ? v : 0.f;
                else if (act == 2) v = gate[at] > 0.f ? v : alpha * v;
            }
            tile[j][threadIdx.x] = v;
            acc[k] += (double)v;
        }
        __syncthreads();
        for (int j = threadIdx.y; j < MF_TILE; j += MF_ROWS) {
            const int p = p0 + j, c = c0 + threadIdx.x;
            if (p < OHW && c < CO) dst[(long long)p * CO + c] = tile[threadIdx.x][j];
        }
        __syncthreads();
    }
    if (!partial) return;
    int k = 0;
    for (int j = threadIdx.y; j < MF_TILE; j += MF_ROWS, ++k) {
        double s = acc[k];
        for (int offset = 16; offset > 0; offset >>= 1) s += __shfl_down_sync(0xffffffffu, s, offset);
        const int c = c0 + j;
        if (threadIdx.x == 0 && c < CO) partial[((long long)n * gridDim.x + blockIdx.x) * CO + c] = s;
    }
}

// out[i] = sum over j = 0..count-1 of partial[j * width + i], summed in order in FP64.
extern "C" __global__ void reduce_partials_f64(const double* partial, float* out, int count, int width) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    double s = 0.0;
    for (int j = 0; j < count; ++j) s += partial[(long long)j * width + i];
    out[i] = (float)s;
}

extern "C" __global__ void add_bias(float* rows, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    rows[idx] += bias[idx % N];
}

extern "C" __global__ void maxpool2d_forward_native(
    const float* x, float* y, int* argmax,
    int N, int C, int H, int W, int OH, int OW, int KH, int KW, int SH, int SW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * OH * OW) return;
    int ow = idx % OW, oh = (idx / OW) % OH, nc = idx / (OW * OH);
    int first = (nc * H + oh * SH) * W + ow * SW;
    int best = first;
    float value = x[first];
    for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            int at = first + kh * W + kw;
            if (x[at] > value) { value = x[at]; best = at; }
        }
    y[idx] = value;
    argmax[idx] = best;
}

extern "C" __global__ void maxpool2d_backward_native(
    const float* dy, const int* argmax, float* dx,
    int N, int C, int H, int W, int OH, int OW, int KH, int KW, int SH, int SW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int iw = idx % W, ih = (idx / W) % H, nc = idx / (W * H);
    int oh0 = max(0, (ih - KH + SH) / SH), oh1 = min(OH - 1, ih / SH);
    int ow0 = max(0, (iw - KW + SW) / SW), ow1 = min(OW - 1, iw / SW);
    float sum = 0.0f;
    for (int oh = oh0; oh <= oh1; ++oh)
        for (int ow = ow0; ow <= ow1; ++ow) {
            int out = (nc * OH + oh) * OW + ow;
            if (argmax[out] == idx) sum += dy[out];
        }
    dx[idx] = sum;
}

)CUDA"  // MSVC caps one string literal at 16 KB; adjacent literals are concatenated
R"CUDA(
// ---- Lecture 19 matmul: C(M,N) = A(M,K) B(K,N), all row-major --------------------------------
// Accumulation modes (the error-free transformations use __f*_rn so they are never contracted):
//   0  sequential FMA, the textbook loop; error grows like sqrt(K) ulp.
//   1  FMA inside groups of G terms, groups combined with an error-free TwoSum; error grows like
//      sqrt(G) ulp independent of K, for one extra TwoSum per group.
//   2  Ogita-Rump-Oishi Dot2: TwoProduct + TwoSum on every term, i.e. the dot product as if
//      accumulated in twice the working precision and rounded once (about 10 flops per term).
template <int ACC> struct Accum;

template <> struct Accum<0> {
    float s = 0.f;
    __device__ __forceinline__ void add(float a, float b) { s = fmaf(a, b, s); }
    __device__ __forceinline__ void group_end() {}
    __device__ __forceinline__ float result() { return s; }
};

template <> struct Accum<1> {
    float s = 0.f, c = 0.f, part = 0.f;
    __device__ __forceinline__ void add(float a, float b) { part = fmaf(a, b, part); }
    __device__ __forceinline__ void group_end() {
        const float t = __fadd_rn(s, part), z = __fsub_rn(t, s);
        c = __fadd_rn(c, __fadd_rn(__fsub_rn(s, __fsub_rn(t, z)), __fsub_rn(part, z)));
        s = t;
        part = 0.f;
    }
    __device__ __forceinline__ float result() { group_end(); return __fadd_rn(s, c); }
};

template <> struct Accum<2> {
    float s = 0.f, c = 0.f;
    __device__ __forceinline__ void add(float a, float b) {
        const float p = __fmul_rn(a, b), e = fmaf(a, b, -p);
        const float t = __fadd_rn(s, p), z = __fsub_rn(t, s);
        c = __fadd_rn(c, __fadd_rn(__fadd_rn(__fsub_rn(s, __fsub_rn(t, z)), __fsub_rn(p, z)), e));
        s = t;
    }
    __device__ __forceinline__ void group_end() {}
    __device__ __forceinline__ float result() { return __fadd_rn(s, c); }
};

// Dot product of a row of A (stride 1) and a column of B (stride N) straight from global memory.
template <int G, int ACC>
__device__ __forceinline__ float dot_global(const float* a, const float* b, int N, int K) {
    Accum<ACC> acc;
    for (int k0 = 0; k0 < K; k0 += G) {
        const int k1 = min(k0 + G, K);
        for (int k = k0; k < k1; ++k) acc.add(a[k], b[(long long)k * N]);
        acc.group_end();
    }
    return acc.result();
}

// Basic version: one thread per output element on a 1-D grid, operands read from global memory.
template <int ACC>
__device__ __forceinline__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int row = idx / N, col = idx % N;
    C[idx] = dot_global<32, ACC>(A + (long long)row * K, B + col, N, K);
}

// Tiled version: the output is cut into TILE x TILE blocks, one 2-D thread block per output
// tile; operands still come straight from global memory (reuse only through L1/L2).
template <int TILE, int ACC>
__device__ __forceinline__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * TILE + threadIdx.y, col = blockIdx.x * TILE + threadIdx.x;
    if (row >= M || col >= N) return;
    C[(long long)row * N + col] = dot_global<TILE, ACC>(A + (long long)row * K, B + col, N, K);
}

// Shared-memory version: TILE x TILE tiles of A and B are staged in __shared__ memory and each
// loaded element is reused TILE times, cutting global traffic by a factor of TILE.
template <int TILE, int ACC>
__device__ __forceinline__ void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    Accum<ACC> acc;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(long long)row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(long long)(k0 + ty) * N + col] : 0.f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc.add(As[ty][k], Bs[k][tx]);
        acc.group_end();
        __syncthreads();
    }
    if (row < M && col < N) C[(long long)row * N + col] = acc.result();
}

#define MF_NAIVE(A) extern "C" __global__ void __launch_bounds__(256) matmul_naive_a##A( \
    const float* a, const float* b, float* c, int M, int N, int K) { matmul_naive<A>(a, b, c, M, N, K); }
#define MF_TILED(T, A) extern "C" __global__ void __launch_bounds__(T * T) matmul_tiled_t##T##_a##A( \
    const float* a, const float* b, float* c, int M, int N, int K) { matmul_tiled<T, A>(a, b, c, M, N, K); }
#define MF_SHARED(T, A) extern "C" __global__ void __launch_bounds__(T * T) matmul_shared_t##T##_a##A( \
    const float* a, const float* b, float* c, int M, int N, int K) { matmul_shared<T, A>(a, b, c, M, N, K); }
MF_NAIVE(0) MF_NAIVE(1) MF_NAIVE(2)
MF_TILED(16, 0) MF_TILED(16, 1) MF_TILED(16, 2) MF_TILED(32, 0) MF_TILED(32, 1) MF_TILED(32, 2)
MF_SHARED(16, 0) MF_SHARED(16, 1) MF_SHARED(16, 2) MF_SHARED(32, 0) MF_SHARED(32, 1) MF_SHARED(32, 2)
)CUDA"  // MSVC caps one string literal at 16 KB; adjacent literals are concatenated
R"CUDA(
// ---- Register-tiled GEMM: C(M,N) = op(A) op(B), row-major ----------------------------------------
// A thread block computes a BM x BN tile of C and each thread a TM x TN sub-tile held in registers,
// made of (TM/4) x (TN/4) quadrants of 4 x 4 spread BM/(TM/4) rows and BN/(TN/4) columns apart, so
// that a warp reads consecutive float4 of the shared tiles (no bank conflicts). Per k a thread loads
// TM + TN floats from shared memory for TM * TN FMAs, where the basic/tiled/shared versions load two
// floats per FMA. The BK-deep tiles of A and B are double buffered: the global loads of the next
// tile are in flight while the current one is consumed.
//   TA: A is stored K x M (element (m, k) at A[k * lda + m]), else M x K; TB: B is stored N x K,
//   else K x N. Every operand is read along its contiguous dimension and transposed into k-major
//   shared tiles where needed; 4 floats of padding per row keep those stores free of bank conflicts.
//   Block z accumulates k in [z * k_chunk, min(K, (z + 1) * k_chunk)) into C + z * c_stride, the
//   split-K partials that gemm_split_reduce sums. Without a split every output accumulates
//   k = 0..K-1 in order, so the results equal the basic kernel bit for bit (kahan groups end every
//   GT tiles, GT * BK = 32 terms as there).
//   VEC reads and writes float4: the contiguous dimension of each operand, N, the leading dimensions
//   and the buffers must be multiples of 4 floats / 16 bytes.
template <int BM, int BN, int BK, int TM, int TN, int ACC, int GT, bool TA, bool TB, bool VEC>
__device__ __forceinline__ void gemm_register(const float* __restrict__ A, const float* __restrict__ B,
                                              float* __restrict__ C, int M, int N, int K, int lda, int ldb,
                                              int ldc, int k_chunk, long long c_stride) {
    constexpr int QM = TM / 4, QN = TN / 4, TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    constexpr int A_VECS = BM * BK / 4, B_VECS = BK * BN / 4;
    constexpr int A_PER = (A_VECS + THREADS - 1) / THREADS, B_PER = (B_VECS + THREADS - 1) / THREADS;
    static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0, "tiles are made of float4");
    __shared__ __align__(16) float As[2][BK][BM + 4];
    __shared__ __align__(16) float Bs[2][BK][BN + 4];
    const int tid = threadIdx.x, tx = tid % TX, ty = tid / TX;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    const int k_begin = blockIdx.z * k_chunk, k_end = min(K, k_begin + k_chunk);
    C += blockIdx.z * c_stride;
    const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 ra[A_PER], rb[B_PER];

    // Elements i..i+3 of a row whose valid length is n: one float4 when VEC, else guarded scalars.
    auto vec4 = [&](const float* row, int i, int n) -> float4 {
        if (VEC) return i < n ? *reinterpret_cast<const float4*>(row + i) : zero;
        return make_float4(i < n ? row[i] : 0.f, i + 1 < n ? row[i + 1] : 0.f,
                           i + 2 < n ? row[i + 2] : 0.f, i + 3 < n ? row[i + 3] : 0.f);
    };
    auto load = [&](int k0) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int v = tid + p * THREADS;
            if (A_VECS % THREADS != 0 && v >= A_VECS) break;
            if (TA) {
                const int k = k0 + v / (BM / 4), m = row0 + (v % (BM / 4)) * 4;
                ra[p] = k < k_end ? vec4(A + (long long)k * lda, m, M) : zero;
            } else {
                const int m = row0 + v / (BK / 4), k = k0 + (v % (BK / 4)) * 4;
                ra[p] = m < M ? vec4(A + (long long)m * lda, k, k_end) : zero;
            }
        }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) {
            const int v = tid + p * THREADS;
            if (B_VECS % THREADS != 0 && v >= B_VECS) break;
            if (TB) {
                const int n = col0 + v / (BK / 4), k = k0 + (v % (BK / 4)) * 4;
                rb[p] = n < N ? vec4(B + (long long)n * ldb, k, k_end) : zero;
            } else {
                const int k = k0 + v / (BN / 4), n = col0 + (v % (BN / 4)) * 4;
                rb[p] = k < k_end ? vec4(B + (long long)k * ldb, n, N) : zero;
            }
        }
    };
    auto store = [&](int buf) {
#pragma unroll
        for (int p = 0; p < A_PER; ++p) {
            const int v = tid + p * THREADS;
            if (A_VECS % THREADS != 0 && v >= A_VECS) break;
            if (TA) {
                *reinterpret_cast<float4*>(&As[buf][v / (BM / 4)][(v % (BM / 4)) * 4]) = ra[p];
            } else {
                const int r = v / (BK / 4), k = (v % (BK / 4)) * 4;
                As[buf][k][r] = ra[p].x; As[buf][k + 1][r] = ra[p].y;
                As[buf][k + 2][r] = ra[p].z; As[buf][k + 3][r] = ra[p].w;
            }
        }
#pragma unroll
        for (int p = 0; p < B_PER; ++p) {
            const int v = tid + p * THREADS;
            if (B_VECS % THREADS != 0 && v >= B_VECS) break;
            if (TB) {
                const int n = v / (BK / 4), k = (v % (BK / 4)) * 4;
                Bs[buf][k][n] = rb[p].x; Bs[buf][k + 1][n] = rb[p].y;
                Bs[buf][k + 2][n] = rb[p].z; Bs[buf][k + 3][n] = rb[p].w;
            } else {
                *reinterpret_cast<float4*>(&Bs[buf][v / (BN / 4)][(v % (BN / 4)) * 4]) = rb[p];
            }
        }
    };

    Accum<ACC> acc[TM][TN];
    const int tiles = k_end > k_begin ? (k_end - k_begin + BK - 1) / BK : 0;
    if (tiles > 0) {
        load(k_begin);
        store(0);
    }
    __syncthreads();
    for (int t = 0; t < tiles; ++t) {
        const int buf = t & 1;
        if (t + 1 < tiles) load(k_begin + (t + 1) * BK);
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
#pragma unroll
            for (int q = 0; q < QM; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&As[buf][k][q * (BM / QM) + ty * 4]);
                a[q * 4] = v.x; a[q * 4 + 1] = v.y; a[q * 4 + 2] = v.z; a[q * 4 + 3] = v.w;
            }
#pragma unroll
            for (int q = 0; q < QN; ++q) {
                const float4 v = *reinterpret_cast<const float4*>(&Bs[buf][k][q * (BN / QN) + tx * 4]);
                b[q * 4] = v.x; b[q * 4 + 1] = v.y; b[q * 4 + 2] = v.z; b[q * 4 + 3] = v.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j].add(a[i], b[j]);
        }
        if ((t + 1) % GT == 0) {
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j].group_end();
        }
        if (t + 1 < tiles) store((t + 1) & 1);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + (i / 4) * (BM / QM) + ty * 4 + i % 4;
        if (r >= M) continue;
#pragma unroll
        for (int q = 0; q < QN; ++q) {
            const int c = col0 + q * (BN / QN) + tx * 4;
            float* dst = C + (long long)r * ldc + c;
            const float x = acc[i][q * 4].result(), y = acc[i][q * 4 + 1].result();
            const float z = acc[i][q * 4 + 2].result(), w = acc[i][q * 4 + 3].result();
            if (VEC && c + 3 < N) {
                *reinterpret_cast<float4*>(dst) = make_float4(x, y, z, w);
            } else {
                if (c < N) dst[0] = x;
                if (c + 1 < N) dst[1] = y;
                if (c + 2 < N) dst[2] = z;
                if (c + 3 < N) dst[3] = w;
            }
        }
    }
}

// Sum of the split-K partials (slices x M x N) in a fixed order, in FP64, rounded once.
extern "C" __global__ void gemm_split_reduce(const float* partial, float* C, int slices, int M, int N, int ldc) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x, total = (long long)M * N;
    if (i >= total) return;
    double s = 0.0;
    for (int z = 0; z < slices; ++z) s += (double)partial[z * total + i];
    C[(i / N) * ldc + i % N] = (float)s;
}

// GEMM engine (fp32, BK = 16): five tile shapes x NN / NT / TN x scalar / float4 access.
#define MF_GEMM(S, BM, BN, TM, TN, T, TA, TB, V) extern "C" __global__ void __launch_bounds__(256) \
    gemm_register_##S##_##T##_v##V(const float* a, const float* b, float* c, int M, int N, int K, int lda, \
                                    int ldb, int ldc, int k_chunk, long long c_stride) { \
    gemm_register<BM, BN, 16, TM, TN, 0, 1, TA, TB, V != 0>(a, b, c, M, N, K, lda, ldb, ldc, k_chunk, c_stride); }
#define MF_GEMM_SHAPE(S, BM, BN, TM, TN) \
    MF_GEMM(S, BM, BN, TM, TN, nn, false, false, 0) MF_GEMM(S, BM, BN, TM, TN, nn, false, false, 1) \
    MF_GEMM(S, BM, BN, TM, TN, nt, false, true, 0) MF_GEMM(S, BM, BN, TM, TN, nt, false, true, 1) \
    MF_GEMM(S, BM, BN, TM, TN, tn, true, false, 0) MF_GEMM(S, BM, BN, TM, TN, tn, true, false, 1)
MF_GEMM_SHAPE(s, 128, 128, 8, 8)
MF_GEMM_SHAPE(h, 128, 64, 8, 4)
MF_GEMM_SHAPE(t, 256, 32, 8, 4)
MF_GEMM_SHAPE(w, 64, 128, 4, 8)
MF_GEMM_SHAPE(x, 32, 256, 4, 8)
// Lecture comparison with compensated summation (128 x 128 tiles, BK = 8: at BK = 16 dot2 runs out of
// registers and slows down 1.5-3x). The fp32 register variant is gemm_register_s_nn without a split.
#define MF_REGISTER(A, V) extern "C" __global__ void __launch_bounds__(256) matmul_register_a##A##_v##V( \
    const float* a, const float* b, float* c, int M, int N, int K) { \
    gemm_register<128, 128, 8, 8, 8, A, 4, false, false, V != 0>(a, b, c, M, N, K, K, N, N, K, 0); }
MF_REGISTER(1, 0) MF_REGISTER(1, 1) MF_REGISTER(2, 0) MF_REGISTER(2, 1)
)CUDA";

// Must list every extern "C" kernel above; indices below refer to this table.
Kernel kernels[] = {
    {"im2col_forward", nullptr},             // 0
    {"col2im_backward", nullptr},            // 1
    {"conv_epilogue_nchw", nullptr},         // 2
    {"conv_grad_rows", nullptr},             // 3
    {"reduce_partials_f64", nullptr},        // 4
    {"add_bias", nullptr},                   // 5
    {"maxpool2d_forward_native", nullptr},   // 6
    {"maxpool2d_backward_native", nullptr},  // 7
    {"matmul_naive_a0", nullptr}, {"matmul_naive_a1", nullptr}, {"matmul_naive_a2", nullptr},  // 8-10
    {"matmul_tiled_t16_a0", nullptr}, {"matmul_tiled_t16_a1", nullptr}, {"matmul_tiled_t16_a2", nullptr},
    {"matmul_tiled_t32_a0", nullptr}, {"matmul_tiled_t32_a1", nullptr}, {"matmul_tiled_t32_a2", nullptr},
    {"matmul_shared_t16_a0", nullptr}, {"matmul_shared_t16_a1", nullptr}, {"matmul_shared_t16_a2", nullptr},
    {"matmul_shared_t32_a0", nullptr}, {"matmul_shared_t32_a1", nullptr}, {"matmul_shared_t32_a2", nullptr},
    {"matmul_register_a1_v0", nullptr}, {"matmul_register_a1_v1", nullptr},  // (accumulate - 1) * 2 + vec
    {"matmul_register_a2_v0", nullptr}, {"matmul_register_a2_v1", nullptr},
    {"gemm_split_reduce", nullptr},
#define MF_GEMM_NAMES(S)                                                                 \
    {"gemm_register_" #S "_nn_v0", nullptr}, {"gemm_register_" #S "_nn_v1", nullptr}, \
    {"gemm_register_" #S "_nt_v0", nullptr}, {"gemm_register_" #S "_nt_v1", nullptr}, \
    {"gemm_register_" #S "_tn_v0", nullptr}, {"gemm_register_" #S "_tn_v1", nullptr},
    MF_GEMM_NAMES(s) MF_GEMM_NAMES(h) MF_GEMM_NAMES(t) MF_GEMM_NAMES(w) MF_GEMM_NAMES(x)  // shape * 6 + layout * 2 + vec
#undef MF_GEMM_NAMES
};
enum KernelId : int {
    K_IM2COL = 0, K_COL2IM, K_EPILOGUE, K_GRAD_ROWS, K_REDUCE, K_ADD_BIAS, K_POOL_FWD, K_POOL_BWD,
    K_MATMUL_NAIVE, K_MATMUL_TILED16 = K_MATMUL_NAIVE + 3, K_MATMUL_TILED32 = K_MATMUL_TILED16 + 3,
    K_MATMUL_SHARED16 = K_MATMUL_TILED32 + 3, K_MATMUL_SHARED32 = K_MATMUL_SHARED16 + 3,
    K_MATMUL_REGISTER = K_MATMUL_SHARED32 + 3, K_SPLIT_REDUCE = K_MATMUL_REGISTER + 4, K_GEMM_REGISTER,
};

// ---------------------------------------------------------------------------------------------
// Error reporting.
// ---------------------------------------------------------------------------------------------
int fail(int status, int line, const std::string& what) {
    last_error = "native_cublas.cpp:" + std::to_string(line) + ": " + what;
    return status;
}

std::string cuda_error_text(CUresult result) {
    const char* name = nullptr;
    const char* text = nullptr;
    if (api.cuGetErrorName) api.cuGetErrorName(result, &name);
    if (api.cuGetErrorString) api.cuGetErrorString(result, &text);
    std::string out = name ? name : ("CUresult " + std::to_string(result));
    if (text) out += std::string(" (") + text + ")";
    return out;
}

std::string cublas_error_text(cublasStatus_t status) {
    const char* name = api.cublasGetStatusName ? api.cublasGetStatusName(status) : nullptr;
    return name ? std::string(name) : "cublasStatus_t " + std::to_string(status);
}

std::string nvrtc_error_text(nvrtcResult result) {
    const char* text = api.nvrtcGetErrorString ? api.nvrtcGetErrorString(result) : nullptr;
    return text ? std::string(text) : "nvrtcResult " + std::to_string(result);
}

#define MF_CU(call)                                                                        \
    do {                                                                                   \
        const CUresult mf_result_ = (call);                                                \
        if (mf_result_ != CUDA_SUCCESS)                                                    \
            return fail(MF_ERR_CUDA, __LINE__, std::string(#call) + ": " + cuda_error_text(mf_result_)); \
    } while (0)

#define MF_CUBLAS(call)                                                                    \
    do {                                                                                   \
        const cublasStatus_t mf_status_ = (call);                                          \
        if (mf_status_ != CUBLAS_STATUS_SUCCESS)                                           \
            return fail(MF_ERR_CUBLAS, __LINE__, std::string(#call) + ": " + cublas_error_text(mf_status_)); \
    } while (0)

#define MF_TRY(call)                                                                       \
    do {                                                                                   \
        const int mf_rc_ = (call);                                                         \
        if (mf_rc_ != MF_OK) return mf_rc_;                                                \
    } while (0)

#define MF_REQUIRE(condition, message)                                                     \
    do {                                                                                   \
        if (!(condition)) return fail(MF_ERR_ARG, __LINE__, message);                      \
    } while (0)

int launch(int id, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
           CUstream stream, void** args, int line) {
    const Kernel& kernel = kernels[id];
    const CUresult result = api.cuLaunchKernel(kernel.function, gx, gy, gz, bx, by, 1, 0, stream, args, nullptr);
    if (result != CUDA_SUCCESS)
        return fail(MF_ERR_CUDA, line, std::string("launch ") + kernel.name + ": " + cuda_error_text(result));
    if (debug_sync) {
        const CUresult synced = api.cuStreamSynchronize(stream);
        if (synced != CUDA_SUCCESS)
            return fail(MF_ERR_CUDA, line, std::string("after kernel ") + kernel.name + ": " + cuda_error_text(synced));
    }
    return MF_OK;
}

unsigned blocks_for(std::int64_t total, unsigned threads) {
    return static_cast<unsigned>((total + threads - 1) / threads);
}

bool fits_index(std::int64_t value) { return value >= 0 && value <= INDEX_LIMIT; }

// ---------------------------------------------------------------------------------------------
// Initialization: symbols, cuBLAS handle, architecture-specific NVRTC build with a disk cache.
// ---------------------------------------------------------------------------------------------
template <typename T>
bool load_symbol(HMODULE module, const char* name, T& target, bool required = true) {
    target = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!target && required) {
        last_error = std::string("missing symbol: ") + name;
        return false;
    }
    return true;
}

std::uint64_t fnv1a(const std::string& text) {
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool read_file(const std::string& path, std::vector<char>& data) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    bool ok = size > 0;
    if (ok) {
        data.resize(static_cast<std::size_t>(size));
        ok = std::fread(data.data(), 1, data.size(), file) == data.size();
    }
    std::fclose(file);
    return ok;
}

// Written under a per-process temporary name and renamed, so concurrent workers never read a
// partially written binary.
void write_file_atomic(const std::string& path, const std::vector<char>& data) {
    const std::string temporary = path + ".tmp" + std::to_string(GetCurrentProcessId());
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (!file) return;
    const bool ok = std::fwrite(data.data(), 1, data.size(), file) == data.size();
    std::fclose(file);
    if (!ok || !MoveFileExA(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING))
        DeleteFileA(temporary.c_str());
}

// The newest NVRTC target that the device can run: SASS for the exact architecture when NVRTC
// supports it (no driver JIT), otherwise PTX for the highest supported lower architecture.
int choose_arch(int device_arch, bool& cubin) {
    int best = device_arch;
    if (api.nvrtcGetNumSupportedArchs && api.nvrtcGetSupportedArchs) {
        int count = 0;
        if (api.nvrtcGetNumSupportedArchs(&count) == NVRTC_SUCCESS && count > 0) {
            std::vector<int> archs(static_cast<std::size_t>(count));
            if (api.nvrtcGetSupportedArchs(archs.data()) == NVRTC_SUCCESS) {
                best = 0;
                for (int arch : archs)
                    if (arch <= device_arch && arch > best) best = arch;
            }
        }
    }
    cubin = best == device_arch && api.nvrtcGetCUBINSize && api.nvrtcGetCUBIN;
    return best;
}

int compile_kernels(const char* cache_dir) {
    CUdevice device = 0;
    int major = 0, minor = 0;
    MF_CU(api.cuCtxGetDevice(&device));
    MF_CU(api.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    MF_CU(api.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    MF_CU(api.cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    const int device_arch = major * 10 + minor;
    bool cubin = false;
    const int arch = choose_arch(device_arch, cubin);
    if (arch <= 0)
        return fail(MF_ERR_INIT, __LINE__, "NVRTC supports no architecture <= sm_" + std::to_string(device_arch));
    int nvrtc_major = 0, nvrtc_minor = 0;
    if (api.nvrtcVersion) api.nvrtcVersion(&nvrtc_major, &nvrtc_minor);

    const std::string arch_option = std::string("--gpu-architecture=") + (cubin ? "sm_" : "compute_") + std::to_string(arch);
    const char* options[] = {"--std=c++17", arch_option.c_str(), "--generate-line-info"};
    const int option_count = static_cast<int>(sizeof(options) / sizeof(options[0]));
    std::string key = kernel_source;
    for (const char* option : options) key += std::string("\n") + option;
    key += "\nnvrtc " + std::to_string(nvrtc_major) + "." + std::to_string(nvrtc_minor);
    char key_hex[17];
    std::snprintf(key_hex, sizeof(key_hex), "%016llx", static_cast<unsigned long long>(fnv1a(key)));
    const std::string cache_path = cache_dir && *cache_dir
        ? std::string(cache_dir) + "\\kernels-" + key_hex + (cubin ? ".cubin" : ".ptx") : std::string();

    const auto started = std::chrono::steady_clock::now();
    std::vector<char> binary;
    const char* cache_state = cache_path.empty() ? "off" : "miss";
    if (cuda_module) {
        cache_state = "loaded";
    } else if (!cache_path.empty() && read_file(cache_path, binary)) {
        if (api.cuModuleLoadData(&cuda_module, binary.data()) == CUDA_SUCCESS) {
            cache_state = "hit";
        } else {
            DeleteFileA(cache_path.c_str());
            binary.clear();
            cuda_module = nullptr;
        }
    }
    if (!cuda_module) {
        nvrtcProgram program = nullptr;
        nvrtcResult rc = api.nvrtcCreateProgram(&program, kernel_source, "myflows_native.cu", 0, nullptr, nullptr);
        if (rc != NVRTC_SUCCESS) return fail(MF_ERR_INIT, __LINE__, "nvrtcCreateProgram: " + nvrtc_error_text(rc));
        rc = api.nvrtcCompileProgram(program, option_count, options);
        if (rc != NVRTC_SUCCESS) {
            std::size_t log_size = 0;
            api.nvrtcGetProgramLogSize(program, &log_size);
            std::vector<char> log(log_size ? log_size : 1, '\0');
            api.nvrtcGetProgramLog(program, log.data());
            api.nvrtcDestroyProgram(&program);
            return fail(MF_ERR_INIT, __LINE__, "NVRTC compile failed (" + arch_option + "): " + log.data());
        }
        std::size_t size = 0;
        rc = cubin ? api.nvrtcGetCUBINSize(program, &size) : api.nvrtcGetPTXSize(program, &size);
        if (rc == NVRTC_SUCCESS && size > 0) {
            binary.assign(size, '\0');
            rc = cubin ? api.nvrtcGetCUBIN(program, binary.data()) : api.nvrtcGetPTX(program, binary.data());
        }
        api.nvrtcDestroyProgram(&program);
        if (rc != NVRTC_SUCCESS || binary.empty())
            return fail(MF_ERR_INIT, __LINE__, "NVRTC returned no binary: " + nvrtc_error_text(rc));
        MF_CU(api.cuModuleLoadData(&cuda_module, binary.data()));
        if (!cache_path.empty()) write_file_atomic(cache_path, binary);
    }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    for (Kernel& kernel : kernels) MF_CU(api.cuModuleGetFunction(&kernel.function, cuda_module, kernel.name));

    char info[512];
    std::snprintf(info, sizeof(info),
                  "{\"device_arch\": %d, \"target\": \"%s_%d\", \"binary\": \"%s\", \"nvrtc\": \"%d.%d\", "
                  "\"cache\": \"%s\", \"load_ms\": %.3f, \"sm_count\": %d, \"cublas_math\": \"CUBLAS_DEFAULT_MATH\"}",
                  device_arch, cubin ? "sm" : "compute", arch, cubin ? "cubin" : "ptx",
                  nvrtc_major, nvrtc_minor, cache_state, load_ms, sm_count);
    info_text = info;
    return MF_OK;
}

int ensure_initialized(const char* cublas_path, const char* nvrtc_path, const char* cache_dir) {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized) return MF_OK;
    driver_module = LoadLibraryA("nvcuda.dll");
    nvrtc_module = LoadLibraryExA(nvrtc_path, nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    cublas_module = LoadLibraryExA(cublas_path, nullptr,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!driver_module || !nvrtc_module || !cublas_module)
        return fail(MF_ERR_INIT, __LINE__, std::string("could not load ") +
                    (!driver_module ? "nvcuda.dll" : !nvrtc_module ? nvrtc_path : cublas_path));
    const bool symbols =
        load_symbol(driver_module, "cuModuleLoadData", api.cuModuleLoadData) &&
        load_symbol(driver_module, "cuModuleGetFunction", api.cuModuleGetFunction) &&
        load_symbol(driver_module, "cuLaunchKernel", api.cuLaunchKernel) &&
        load_symbol(driver_module, "cuStreamSynchronize", api.cuStreamSynchronize) &&
        load_symbol(driver_module, "cuCtxGetCurrent", api.cuCtxGetCurrent) &&
        load_symbol(driver_module, "cuCtxGetDevice", api.cuCtxGetDevice) &&
        load_symbol(driver_module, "cuDeviceGetAttribute", api.cuDeviceGetAttribute) &&
        load_symbol(driver_module, "cuGetErrorName", api.cuGetErrorName, false) &&
        load_symbol(driver_module, "cuGetErrorString", api.cuGetErrorString, false) &&
        load_symbol(nvrtc_module, "nvrtcCreateProgram", api.nvrtcCreateProgram) &&
        load_symbol(nvrtc_module, "nvrtcCompileProgram", api.nvrtcCompileProgram) &&
        load_symbol(nvrtc_module, "nvrtcGetProgramLogSize", api.nvrtcGetProgramLogSize) &&
        load_symbol(nvrtc_module, "nvrtcGetProgramLog", api.nvrtcGetProgramLog) &&
        load_symbol(nvrtc_module, "nvrtcGetPTXSize", api.nvrtcGetPTXSize) &&
        load_symbol(nvrtc_module, "nvrtcGetPTX", api.nvrtcGetPTX) &&
        load_symbol(nvrtc_module, "nvrtcGetCUBINSize", api.nvrtcGetCUBINSize, false) &&
        load_symbol(nvrtc_module, "nvrtcGetCUBIN", api.nvrtcGetCUBIN, false) &&
        load_symbol(nvrtc_module, "nvrtcDestroyProgram", api.nvrtcDestroyProgram) &&
        load_symbol(nvrtc_module, "nvrtcVersion", api.nvrtcVersion, false) &&
        load_symbol(nvrtc_module, "nvrtcGetNumSupportedArchs", api.nvrtcGetNumSupportedArchs, false) &&
        load_symbol(nvrtc_module, "nvrtcGetSupportedArchs", api.nvrtcGetSupportedArchs, false) &&
        load_symbol(nvrtc_module, "nvrtcGetErrorString", api.nvrtcGetErrorString, false) &&
        load_symbol(cublas_module, "cublasCreate_v2", api.cublasCreate) &&
        load_symbol(cublas_module, "cublasSetStream_v2", api.cublasSetStream) &&
        load_symbol(cublas_module, "cublasSetMathMode", api.cublasSetMathMode) &&
        load_symbol(cublas_module, "cublasSgemm_v2", api.cublasSgemm) &&
        load_symbol(cublas_module, "cublasGetStatusName", api.cublasGetStatusName, false);
    if (!symbols) return fail(MF_ERR_INIT, __LINE__, last_error);
    MF_CU(api.cuCtxGetCurrent(&init_context));
    if (!init_context)
        return fail(MF_ERR_INIT, __LINE__, "no current CUDA context; create a CuPy array before initializing");
    char sync_flag[8] = {0};
    if (GetEnvironmentVariableA("MYFLOWS_NATIVE_SYNC", sync_flag, sizeof(sync_flag)) > 0)
        debug_sync = sync_flag[0] != '0';
    // A failed earlier attempt may already own the handle or the module; reuse them on retry.
    if (!cublas_handle) MF_CUBLAS(api.cublasCreate(&cublas_handle));
    MF_CUBLAS(api.cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    MF_TRY(compile_kernels(cache_dir));
    initialized = true;
    return MF_OK;
}

// Every entry point runs on the CUDA context that the module and the cuBLAS handle belong to.
int check_ready() {
    if (!initialized) return fail(MF_ERR_INIT, __LINE__, "mf_init has not succeeded");
    CUcontext current = nullptr;
    MF_CU(api.cuCtxGetCurrent(&current));
    if (current != init_context)
        return fail(MF_ERR_CONTEXT, __LINE__, "called on a different CUDA context than the one used by mf_init");
    return MF_OK;
}

// Row-major C(m x n) = op(A) op(B) expressed through cuBLAS' column-major convention by the caller.
int gemm(CUstream stream, int trans_a, int trans_b, int m, int n, int k,
         CUdeviceptr a, int lda, CUdeviceptr b, int ldb, CUdeviceptr c, int ldc, int line) {
    const cublasStatus_t bound = api.cublasSetStream(cublas_handle, stream);
    if (bound != CUBLAS_STATUS_SUCCESS)
        return fail(MF_ERR_CUBLAS, line, "cublasSetStream: " + cublas_error_text(bound));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = api.cublasSgemm(
        cublas_handle, trans_a, trans_b, m, n, k, &alpha,
        reinterpret_cast<const float*>(a), lda, reinterpret_cast<const float*>(b), ldb, &beta,
        reinterpret_cast<float*>(c), ldc);
    if (status != CUBLAS_STATUS_SUCCESS)
        return fail(MF_ERR_CUBLAS, line, "cublasSgemm: " + cublas_error_text(status));
    if (debug_sync) {
        const CUresult synced = api.cuStreamSynchronize(stream);
        if (synced != CUDA_SUCCESS) return fail(MF_ERR_CUDA, line, "after cublasSgemm: " + cuda_error_text(synced));
    }
    return MF_OK;
}

// Register-tiled GEMM engine. Tile shapes in kernel order s, h, t, w, x (BM x BN, 256 threads each):
// narrow (N <= 64) and short (M <= 128) outputs use tiles stretched along their long side, so that
// fewer of the computed outputs fall outside C.
enum GemmEngine : int { ENGINE_CUBLAS = 0, ENGINE_REGISTER = 1 };
struct GemmShape {
    int bm, bn;
};
constexpr GemmShape GEMM_SHAPES[] = {{128, 128}, {128, 64}, {256, 32}, {64, 128}, {32, 256}};
constexpr int GEMM_BK = 16;
constexpr int GEMM_MIN_CHUNK = 64;
constexpr int GEMM_MAX_SLICES = 32;

struct GemmPlan {
    int shape, slices, k_chunk;
};

// An output of fewer than 3 tiles per SM leaves most of the GPU idle (an SM holds 2-3 blocks). When
// K is at least M + N, K is then split into slices of at least GEMM_MIN_CHUNK (whole BK tiles)
// until about 6 tiles per SM run; each slice writes a partial product and gemm_split_reduce sums
// them. For shorter K those partial sums cost more memory traffic than the split gains (measured on
// the ResNet18 convolution GEMMs: forward GEMMs ran 1.6-2.2x slower split); the weight gradients,
// whose K is the batch's pixel count, are the outputs that need it.
GemmPlan gemm_plan(int M, int N, int K) {
    const int shape = N <= 32 ? 2 : N <= 64 ? 1 : M <= 32 ? 4 : M <= 64 ? 3 : M <= 128 ? 1 : 0;
    GemmPlan plan{shape, 1, K};
    const std::int64_t tiles = static_cast<std::int64_t>(blocks_for(M, GEMM_SHAPES[shape].bm)) *
                               blocks_for(N, GEMM_SHAPES[shape].bn);
    const std::int64_t target = 6LL * sm_count;
    if (tiles * 2 >= target || K < static_cast<std::int64_t>(M) + N) return plan;
    const std::int64_t split = std::min<std::int64_t>({(target + tiles - 1) / tiles, K / GEMM_MIN_CHUNK, GEMM_MAX_SLICES});
    if (split < 2) return plan;
    plan.k_chunk = static_cast<int>(((K + split - 1) / split + GEMM_BK - 1) / GEMM_BK * GEMM_BK);
    plan.slices = (K + plan.k_chunk - 1) / plan.k_chunk;
    return plan;
}

std::int64_t gemm_workspace_floats(int M, int N, int K) {
    const GemmPlan plan = gemm_plan(M, N, K);
    return plan.slices > 1 ? static_cast<std::int64_t>(plan.slices) * M * N : 0;
}

// Row-major C(M x N) = op(A) op(B): A is stored K x M when trans_a, else M x K; B is stored N x K
// when trans_b, else K x N. The register engine splits K only when the workspace (CuPy-owned, may
// be 0) holds gemm_workspace_floats; the split sums each output in a different order than one pass.
int gemm_rows(int engine, CUstream stream, bool trans_a, bool trans_b, int M, int N, int K,
              CUdeviceptr a, int lda, CUdeviceptr b, int ldb, CUdeviceptr c, int ldc,
              CUdeviceptr workspace, std::int64_t workspace_floats, int line) {
    if (engine == ENGINE_CUBLAS)  // column-major C^T = op(B)^T op(A)^T
        return gemm(stream, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                    N, M, K, b, ldb, a, lda, c, ldc, line);
    if (trans_a && trans_b) return fail(MF_ERR_ARG, line, "the register engine has no A^T B^T kernel");
    GemmPlan plan = gemm_plan(M, N, K);
    if (plan.slices > 1 && (!workspace || workspace_floats < static_cast<std::int64_t>(plan.slices) * M * N))
        plan = GemmPlan{plan.shape, 1, K};
    const bool split = plan.slices > 1;
    CUdeviceptr out = split ? workspace : c;
    int ld_out = split ? N : ldc;
    long long stride = split ? static_cast<long long>(M) * N : 0;
    // float4 access needs the contiguous dimension of every operand, N, the leading dimensions and
    // the buffers to be multiples of 4 floats / 16 bytes.
    const int vec = (trans_a ? M : K) % 4 == 0 && (trans_b ? K : N) % 4 == 0 && N % 4 == 0 &&
                    lda % 4 == 0 && ldb % 4 == 0 && ld_out % 4 == 0 && ((a | b | out) & 15) == 0;
    const GemmShape& tile = GEMM_SHAPES[plan.shape];
    if (blocks_for(M, tile.bm) > GRID_YZ_LIMIT) return fail(MF_ERR_ARG, line, "GEMM rows exceed the CUDA grid limit");
    const int layout = trans_a ? 2 : trans_b ? 1 : 0;  // nn, nt, tn
    {
        void* args[] = {&a, &b, &out, &M, &N, &K, &lda, &ldb, &ld_out, &plan.k_chunk, &stride};
        MF_TRY(launch(K_GEMM_REGISTER + plan.shape * 6 + layout * 2 + vec, blocks_for(N, tile.bn),
                      blocks_for(M, tile.bm), static_cast<unsigned>(plan.slices), 256, 1, stream, args, line));
    }
    if (!split) return MF_OK;
    void* args[] = {&workspace, &c, &plan.slices, &M, &N, &ldc};
    return launch(K_SPLIT_REDUCE, blocks_for(static_cast<std::int64_t>(M) * N, 256), 1, 1, 256, 1, stream, args, line);
}

struct ConvDims {
    int N, CI, H, W, CO, KH, KW, OH, OW, SH, SW, PH, PW;
};

int check_conv(const ConvDims& d, int line) {
    const bool positive = d.N > 0 && d.CI > 0 && d.H > 0 && d.W > 0 && d.CO > 0 && d.KH > 0 &&
                          d.KW > 0 && d.OH > 0 && d.OW > 0 && d.SH > 0 && d.SW > 0 && d.PH >= 0 && d.PW >= 0;
    if (!positive) return fail(MF_ERR_ARG, line, "convolution dimensions must be positive");
    if (d.OH != (d.H + 2 * d.PH - d.KH) / d.SH + 1 || d.OW != (d.W + 2 * d.PW - d.KW) / d.SW + 1)
        return fail(MF_ERR_ARG, line, "convolution output size does not match input, kernel, stride and padding");
    const std::int64_t rows = static_cast<std::int64_t>(d.N) * d.OH * d.OW;
    const std::int64_t features = static_cast<std::int64_t>(d.CI) * d.KH * d.KW;
    if (!fits_index(rows * features) || !fits_index(rows * d.CO) ||
        !fits_index(static_cast<std::int64_t>(d.N) * d.CI * d.H * d.W) || !fits_index(features * d.CO))
        return fail(MF_ERR_ARG, line, "convolution buffers exceed the int32 indexing limit");
    if (static_cast<unsigned>(d.N) > GRID_YZ_LIMIT || static_cast<unsigned>((d.CO + 31) / 32) > GRID_YZ_LIMIT)
        return fail(MF_ERR_ARG, line, "batch or channel count exceeds the CUDA grid limit");
    return MF_OK;
}

int check_engine(int engine, int line) {
    if (engine != ENGINE_CUBLAS && engine != ENGINE_REGISTER)
        return fail(MF_ERR_ARG, line, "GEMM engine must be 0 (cuBLAS) or 1 (register-tiled)");
    return MF_OK;
}

int check_activation(int activation, float alpha, int line) {
    if (activation != ACT_NONE && activation != ACT_RELU && activation != ACT_LEAKY_RELU)
        return fail(MF_ERR_ARG, line, "unknown activation code");
    if (activation == ACT_LEAKY_RELU && !(alpha > 0.0f))
        return fail(MF_ERR_ARG, line, "fused LeakyReLU requires alpha > 0");
    return MF_OK;
}

}  // namespace

extern "C" __declspec(dllexport) const char* mf_last_error() {
    return last_error.c_str();
}

extern "C" __declspec(dllexport) const char* mf_info() {
    return info_text.c_str();
}

extern "C" __declspec(dllexport) void mf_set_debug_sync(int enabled) {
    debug_sync = enabled != 0;
}

extern "C" __declspec(dllexport) int mf_init(const char* cublas_path, const char* nvrtc_path, const char* cache_dir) {
    return ensure_initialized(cublas_path, nvrtc_path, cache_dir);
}

// Forward: im2col, rows = cols W^T (GEMM engine), then the NCHW epilogue (bias + activation).
// engine: 0 cuBLAS, 1 register-tiled, with an optional split-K workspace (see mf_gemm).
extern "C" __declspec(dllexport) int mf_conv_forward(
    std::uint64_t x, std::uint64_t weight, std::uint64_t bias, std::uint64_t cols,
    std::uint64_t rows, std::uint64_t y, int N, int CI, int H, int W, int CO, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW, int activation, float alpha, int engine,
    std::uint64_t workspace, std::int64_t workspace_floats, std::uint64_t stream_ptr) {
    MF_TRY(check_ready());
    const ConvDims d{N, CI, H, W, CO, KH, KW, OH, OW, SH, SW, PH, PW};
    MF_TRY(check_conv(d, __LINE__));
    MF_TRY(check_activation(activation, alpha, __LINE__));
    MF_TRY(check_engine(engine, __LINE__));
    MF_REQUIRE(x && weight && cols && rows && y, "conv forward received a null buffer");
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    const int M = N * OH * OW;
    const int K = CI * KH * KW;
    int OHW = OH * OW;
    {
        void* args[] = {&x, &cols, &N, &CI, &H, &W, &KH, &KW, &OH, &OW, &SH, &SW, &PH, &PW};
        MF_TRY(launch(K_IM2COL, blocks_for(static_cast<std::int64_t>(M) * K, 256), 1, 1, 256, 1, stream, args, __LINE__));
    }
    MF_TRY(gemm_rows(engine, stream, false, true, M, CO, K, cols, K, weight, K, rows, CO, workspace,
                     workspace_floats, __LINE__));
    void* args[] = {&rows, &bias, &y, &OHW, &CO, &activation, &alpha};
    return launch(K_EPILOGUE, blocks_for(OHW, 32), blocks_for(CO, 32), N, 32, 8, stream, args, __LINE__);
}

// Backward: dyT = gate(dy) transposed to (M, CO) with FP64 bias partials, dW = dyT^T cols,
// grad_cols = dyT W, dx = col2im(grad_cols). dx/grad_cols and partial/db are optional (0).
extern "C" __declspec(dllexport) int mf_conv_backward(
    std::uint64_t weight, std::uint64_t dy, std::uint64_t y, std::uint64_t cols,
    std::uint64_t dyT, std::uint64_t dw, std::uint64_t grad_cols, std::uint64_t dx,
    std::uint64_t partial, std::uint64_t db, int N, int CI, int H, int W, int CO, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW, int grid_x, int activation, float alpha,
    int engine, std::uint64_t workspace, std::int64_t workspace_floats, std::uint64_t stream_ptr) {
    MF_TRY(check_ready());
    const ConvDims d{N, CI, H, W, CO, KH, KW, OH, OW, SH, SW, PH, PW};
    MF_TRY(check_conv(d, __LINE__));
    MF_TRY(check_activation(activation, alpha, __LINE__));
    MF_TRY(check_engine(engine, __LINE__));
    MF_REQUIRE(weight && dy && cols && dyT && dw, "conv backward received a null buffer");
    MF_REQUIRE(activation == ACT_NONE || y, "a fused activation needs the forward output");
    MF_REQUIRE((grad_cols == 0) == (dx == 0), "grad_cols and dx must be given together");
    MF_REQUIRE((partial == 0) == (db == 0), "partial and db must be given together");
    int OHW = OH * OW;
    MF_REQUIRE(grid_x > 0 && grid_x <= (OHW + 31) / 32, "grid_x must be in [1, spatial tiles]");
    MF_REQUIRE(fits_index(static_cast<std::int64_t>(N) * grid_x * CO), "bias partial buffer exceeds int32 indexing");
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    const int M = N * OHW;
    const int K = CI * KH * KW;
    {
        void* args[] = {&dy, &y, &dyT, &partial, &OHW, &CO, &activation, &alpha};
        MF_TRY(launch(K_GRAD_ROWS, static_cast<unsigned>(grid_x), blocks_for(CO, 32), N, 32, 8, stream, args, __LINE__));
    }
    if (db) {
        int count = N * grid_x;
        void* args[] = {&partial, &db, &count, &CO};
        MF_TRY(launch(K_REDUCE, blocks_for(CO, 128), 1, 1, 128, 1, stream, args, __LINE__));
    }
    // dW (CO x K) = dyT^T cols, grad_cols (M x K) = dyT W; both reuse the workspace in stream order.
    MF_TRY(gemm_rows(engine, stream, true, false, CO, K, M, dyT, CO, cols, K, dw, K, workspace,
                     workspace_floats, __LINE__));
    if (!dx) return MF_OK;
    MF_TRY(gemm_rows(engine, stream, false, false, M, K, CO, dyT, CO, weight, K, grad_cols, K, workspace,
                     workspace_floats, __LINE__));
    void* args[] = {&grad_cols, &dx, &N, &CI, &H, &W, &KH, &KW, &OH, &OW, &SH, &SW, &PH, &PW};
    return launch(K_COL2IM, blocks_for(static_cast<std::int64_t>(N) * CI * H * W, 256), 1, 1, 256, 1, stream, args, __LINE__);
}

extern "C" __declspec(dllexport) int mf_pool_forward(
    std::uint64_t x, std::uint64_t y, std::uint64_t argmax,
    int N, int C, int H, int W, int OH, int OW, int KH, int KW, int SH, int SW,
    std::uint64_t stream_ptr) {
    MF_TRY(check_ready());
    MF_REQUIRE(N > 0 && C > 0 && H > 0 && W > 0 && OH > 0 && OW > 0 && KH > 0 && KW > 0 && SH > 0 && SW > 0,
               "pooling dimensions must be positive");
    MF_REQUIRE(OH == (H - KH) / SH + 1 && OW == (W - KW) / SW + 1, "pooling output size does not match the input");
    MF_REQUIRE(fits_index(static_cast<std::int64_t>(N) * C * H * W), "pooling input exceeds int32 indexing");
    MF_REQUIRE(x && y && argmax, "pool forward received a null buffer");
    void* args[] = {&x, &y, &argmax, &N, &C, &H, &W, &OH, &OW, &KH, &KW, &SH, &SW};
    return launch(K_POOL_FWD, blocks_for(static_cast<std::int64_t>(N) * C * OH * OW, 256), 1, 1, 256, 1,
                  reinterpret_cast<CUstream>(stream_ptr), args, __LINE__);
}

extern "C" __declspec(dllexport) int mf_pool_backward(
    std::uint64_t dy, std::uint64_t argmax, std::uint64_t dx,
    int N, int C, int H, int W, int OH, int OW, int KH, int KW, int SH, int SW,
    std::uint64_t stream_ptr) {
    MF_TRY(check_ready());
    MF_REQUIRE(N > 0 && C > 0 && H > 0 && W > 0 && OH > 0 && OW > 0 && KH > 0 && KW > 0 && SH > 0 && SW > 0,
               "pooling dimensions must be positive");
    MF_REQUIRE(OH == (H - KH) / SH + 1 && OW == (W - KW) / SW + 1, "pooling output size does not match the input");
    MF_REQUIRE(fits_index(static_cast<std::int64_t>(N) * C * H * W), "pooling input exceeds int32 indexing");
    MF_REQUIRE(dy && argmax && dx, "pool backward received a null buffer");
    void* args[] = {&dy, &argmax, &dx, &N, &C, &H, &W, &OH, &OW, &KH, &KW, &SH, &SW};
    return launch(K_POOL_BWD, blocks_for(static_cast<std::int64_t>(N) * C * H * W, 256), 1, 1, 256, 1,
                  reinterpret_cast<CUstream>(stream_ptr), args, __LINE__);
}

// C(M,N) = A(M,K) B(K,N) [+ bias(N)], row-major. variant: 0 basic, 1 tiled, 2 shared memory,
// 3 cuBLAS, 4 register-tiled; tile 16/32 for variants 1-2; accumulate 0/1/2 as documented in the
// kernel source.
extern "C" __declspec(dllexport) int mf_matmul(
    int variant, int tile, int accumulate, std::uint64_t a, std::uint64_t b, std::uint64_t bias,
    std::uint64_t c, int M, int N, int K, std::uint64_t stream_ptr) {
    MF_TRY(check_ready());
    MF_REQUIRE(M > 0 && N > 0 && K > 0, "matmul dimensions must be positive");
    MF_REQUIRE(fits_index(static_cast<std::int64_t>(M) * K) && fits_index(static_cast<std::int64_t>(K) * N) &&
               fits_index(static_cast<std::int64_t>(M) * N), "matmul operands exceed int32 indexing");
    MF_REQUIRE(a && b && c, "matmul received a null buffer");
    MF_REQUIRE(variant >= 0 && variant <= 4,
               "matmul variant must be 0 (basic), 1 (tiled), 2 (shared), 3 (cuBLAS) or 4 (register-tiled)");
    MF_REQUIRE(accumulate >= 0 && accumulate <= 2, "accumulate must be 0, 1 or 2");
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    if (variant == 3) {
        // Row-major C = A B is column-major C^T = B^T A^T.
        MF_TRY(gemm(stream, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, b, N, a, K, c, N, __LINE__));
    } else {
        void* args[] = {&a, &b, &c, &M, &N, &K};
        if (variant == 0) {
            MF_TRY(launch(K_MATMUL_NAIVE + accumulate, blocks_for(static_cast<std::int64_t>(M) * N, 256), 1, 1,
                          256, 1, stream, args, __LINE__));
        } else if (variant == 4) {
            // 128 x 128 tiles, one pass over K: fp32 is the GEMM engine's kernel (BK = 16), kahan and
            // dot2 the BK = 8 lecture kernels. float4 access needs K and N multiples of 4 and aligned buffers.
            const int vec = K % 4 == 0 && N % 4 == 0 && ((a | b | c) & 15) == 0;
            MF_REQUIRE(blocks_for(M, 128) <= GRID_YZ_LIMIT, "matmul rows exceed the CUDA grid limit");
            if (accumulate == 0) {
                long long stride = 0;
                void* engine_args[] = {&a, &b, &c, &M, &N, &K, &K, &N, &N, &K, &stride};
                MF_TRY(launch(K_GEMM_REGISTER + vec, blocks_for(N, 128), blocks_for(M, 128), 1, 256, 1, stream,
                              engine_args, __LINE__));
            } else {
                MF_TRY(launch(K_MATMUL_REGISTER + (accumulate - 1) * 2 + vec, blocks_for(N, 128),
                              blocks_for(M, 128), 1, 256, 1, stream, args, __LINE__));
            }
        } else {
            MF_REQUIRE(tile == 16 || tile == 32, "tile must be 16 or 32");
            MF_REQUIRE(blocks_for(M, tile) <= GRID_YZ_LIMIT, "matmul rows exceed the CUDA grid limit");
            const int base = variant == 1 ? (tile == 16 ? K_MATMUL_TILED16 : K_MATMUL_TILED32)
                                          : (tile == 16 ? K_MATMUL_SHARED16 : K_MATMUL_SHARED32);
            MF_TRY(launch(base + accumulate, blocks_for(N, tile), blocks_for(M, tile), 1,
                          static_cast<unsigned>(tile), static_cast<unsigned>(tile), stream, args, __LINE__));
        }
    }
    if (!bias) return MF_OK;
    void* args[] = {&c, &bias, &M, &N};
    return launch(K_ADD_BIAS, blocks_for(static_cast<std::int64_t>(M) * N, 256), 1, 1, 256, 1, stream, args, __LINE__);
}

// Tile shape (index into s, h, t, w, x) and split-K slices of the register engine for C(M,N) with
// inner size K.
extern "C" __declspec(dllexport) int mf_gemm_plan(int M, int N, int K, int* shape, int* slices) {
    MF_TRY(check_ready());
    MF_REQUIRE(M > 0 && N > 0 && K > 0, "GEMM dimensions must be positive");
    MF_REQUIRE(shape && slices, "mf_gemm_plan received a null output");
    const GemmPlan plan = gemm_plan(M, N, K);
    *shape = plan.shape;
    *slices = plan.slices;
    return MF_OK;
}

// Floats of split-K workspace that the register engine of mf_gemm uses for C(M,N) with inner size
// K (0 when it does not split), or -1 with mf_last_error set.
extern "C" __declspec(dllexport) std::int64_t mf_gemm_workspace(int M, int N, int K) {
    if (check_ready() != MF_OK) return -1;
    if (M <= 0 || N <= 0 || K <= 0) {
        fail(MF_ERR_ARG, __LINE__, "GEMM dimensions must be positive");
        return -1;
    }
    return gemm_workspace_floats(M, N, K);
}

// C(M,N) = op(A) op(B) [+ bias(N)], row-major with leading dimensions lda/ldb/ldc. op(A) = A^T
// when trans_a (A stored K x M), op(B) = B^T when trans_b (B stored N x K); not both. engine: 0
// cuBLAS, 1 register-tiled. workspace may be 0; with mf_gemm_workspace(M, N, K) floats the register
// engine splits K for outputs too small to fill the GPU. bias needs ldc == N.
extern "C" __declspec(dllexport) int mf_gemm(
    int engine, int trans_a, int trans_b, std::uint64_t a, std::uint64_t b, std::uint64_t bias,
    std::uint64_t c, int M, int N, int K, int lda, int ldb, int ldc, std::uint64_t workspace,
    std::int64_t workspace_floats, std::uint64_t stream_ptr) {
    MF_TRY(check_ready());
    MF_TRY(check_engine(engine, __LINE__));
    MF_REQUIRE(M > 0 && N > 0 && K > 0, "GEMM dimensions must be positive");
    MF_REQUIRE(!(trans_a && trans_b), "GEMM supports A^T B and A B^T but not A^T B^T");
    MF_REQUIRE(lda >= (trans_a ? M : K) && ldb >= (trans_b ? K : N) && ldc >= N,
               "GEMM leading dimensions are smaller than the rows");
    MF_REQUIRE(fits_index(static_cast<std::int64_t>(M) * K) && fits_index(static_cast<std::int64_t>(K) * N) &&
               fits_index(static_cast<std::int64_t>(M) * N), "GEMM operands exceed int32 indexing");
    MF_REQUIRE(a && b && c, "GEMM received a null buffer");
    MF_REQUIRE(!bias || ldc == N, "GEMM bias needs ldc == N");
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    MF_TRY(gemm_rows(engine, stream, trans_a != 0, trans_b != 0, M, N, K, a, lda, b, ldb, c, ldc, workspace,
                     workspace_floats, __LINE__));
    if (!bias) return MF_OK;
    void* args[] = {&c, &bias, &M, &N};
    return launch(K_ADD_BIAS, blocks_for(static_cast<std::int64_t>(M) * N, 256), 1, 1, 256, 1, stream, args, __LINE__);
}
