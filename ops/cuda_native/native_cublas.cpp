#include <windows.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace {

using CUresult = int;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = std::uint64_t;
using nvrtcResult = int;
using nvrtcProgram = void*;
using cublasHandle_t = void*;
using cublasStatus_t = int;

constexpr int CUDA_SUCCESS = 0;
constexpr int NVRTC_SUCCESS = 0;
constexpr int CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_OP_N = 0;
constexpr int CUBLAS_OP_T = 1;

using CuModuleLoadData = CUresult (*)(CUmodule*, const void*);
using CuModuleGetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuLaunchKernel = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                    unsigned, unsigned, unsigned, unsigned,
                                    CUstream, void**, void**);
using NvrtcCreateProgram = nvrtcResult (*)(nvrtcProgram*, const char*, const char*, int,
                                           const char* const*, const char* const*);
using NvrtcCompileProgram = nvrtcResult (*)(nvrtcProgram, int, const char* const*);
using NvrtcGetProgramLogSize = nvrtcResult (*)(nvrtcProgram, std::size_t*);
using NvrtcGetProgramLog = nvrtcResult (*)(nvrtcProgram, char*);
using NvrtcGetPtxSize = nvrtcResult (*)(nvrtcProgram, std::size_t*);
using NvrtcGetPtx = nvrtcResult (*)(nvrtcProgram, char*);
using NvrtcDestroyProgram = nvrtcResult (*)(nvrtcProgram*);
using CublasCreate = cublasStatus_t (*)(cublasHandle_t*);
using CublasDestroy = cublasStatus_t (*)(cublasHandle_t);
using CublasSetStream = cublasStatus_t (*)(cublasHandle_t, CUstream);
using CublasSgemm = cublasStatus_t (*)(cublasHandle_t, int, int, int, int, int,
                                       const float*, const float*, int,
                                       const float*, int, const float*, float*, int);

HMODULE driver_module = nullptr;
HMODULE nvrtc_module = nullptr;
HMODULE cublas_module = nullptr;
CUmodule cuda_module = nullptr;
cublasHandle_t cublas_handle = nullptr;
CuModuleLoadData cuModuleLoadData = nullptr;
CuModuleGetFunction cuModuleGetFunction = nullptr;
CuLaunchKernel cuLaunchKernel = nullptr;
CublasCreate cublasCreate = nullptr;
CublasDestroy cublasDestroy = nullptr;
CublasSetStream cublasSetStream = nullptr;
CublasSgemm cublasSgemm = nullptr;
NvrtcCreateProgram nvrtcCreateProgram = nullptr;
NvrtcCompileProgram nvrtcCompileProgram = nullptr;
NvrtcGetProgramLogSize nvrtcGetProgramLogSize = nullptr;
NvrtcGetProgramLog nvrtcGetProgramLog = nullptr;
NvrtcGetPtxSize nvrtcGetPtxSize = nullptr;
NvrtcGetPtx nvrtcGetPtx = nullptr;
NvrtcDestroyProgram nvrtcDestroyProgram = nullptr;
CUfunction im2col_function = nullptr;
CUfunction col2im_function = nullptr;
CUfunction add_bias_function = nullptr;
CUfunction bias_grad_function = nullptr;
std::mutex init_mutex;
bool initialized = false;
std::string last_error;

const char* kernel_source = R"CUDA(
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

extern "C" __global__ void add_bias(float* rows, const float* bias, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    rows[idx] += bias[idx % N];
}

extern "C" __global__ void bias_grad_rows(
    const float* dy, float* db, int M, int CO) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= CO) return;
    float sum = 0.0f;
    for (int row = 0; row < M; ++row)
        sum += dy[row * CO + channel];
    db[channel] = sum;
}
)CUDA";

template <typename T>
bool load_symbol(HMODULE module, const char* name, T& target) {
    target = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!target) {
        last_error = std::string("missing symbol: ") + name;
        return false;
    }
    return true;
}

bool compile_kernels() {
    const char* options[] = {"--std=c++11", "--gpu-architecture=compute_89"};
    nvrtcProgram program = nullptr;
    if (nvrtcCreateProgram(&program, kernel_source, "myflows_native.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
        return false;
    nvrtcResult result = nvrtcCompileProgram(program, 2, options);
    if (result != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::vector<char> log(log_size ? log_size : 1, '\0');
        nvrtcGetProgramLog(program, log.data());
        last_error = std::string("NVRTC compile failed: ") + log.data();
        nvrtcDestroyProgram(&program);
        return false;
    }
    std::size_t ptx_size = 0;
    if (nvrtcGetPtxSize(program, &ptx_size) != NVRTC_SUCCESS) {
        last_error = "NVRTC could not return PTX size";
        nvrtcDestroyProgram(&program);
        return false;
    }
    std::vector<char> ptx(ptx_size ? ptx_size : 1, '\0');
    if (nvrtcGetPtx(program, ptx.data()) != NVRTC_SUCCESS) {
        last_error = "NVRTC could not return PTX";
        nvrtcDestroyProgram(&program);
        return false;
    }
    nvrtcDestroyProgram(&program);
    if (cuModuleLoadData(&cuda_module, ptx.data()) != CUDA_SUCCESS) {
        last_error = "CUDA driver could not load generated PTX";
        return false;
    }
    return cuModuleGetFunction(&im2col_function, cuda_module, "im2col_forward") == CUDA_SUCCESS &&
           cuModuleGetFunction(&col2im_function, cuda_module, "col2im_backward") == CUDA_SUCCESS &&
           cuModuleGetFunction(&add_bias_function, cuda_module, "add_bias") == CUDA_SUCCESS &&
           cuModuleGetFunction(&bias_grad_function, cuda_module, "bias_grad_rows") == CUDA_SUCCESS;
}

bool ensure_initialized(const char* cublas_path, const char* nvrtc_path) {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized) return true;
    driver_module = LoadLibraryA("nvcuda.dll");
    nvrtc_module = LoadLibraryExA(nvrtc_path, nullptr,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    cublas_module = LoadLibraryExA(cublas_path, nullptr,
                                   LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!driver_module || !nvrtc_module || !cublas_module) {
        last_error = "could not load CUDA driver, NVRTC, or cuBLAS DLL";
        return false;
    }
    if (!load_symbol(driver_module, "cuModuleLoadData", cuModuleLoadData) ||
        !load_symbol(driver_module, "cuModuleGetFunction", cuModuleGetFunction) ||
        !load_symbol(driver_module, "cuLaunchKernel", cuLaunchKernel) ||
        !load_symbol(nvrtc_module, "nvrtcCreateProgram", nvrtcCreateProgram) ||
        !load_symbol(nvrtc_module, "nvrtcCompileProgram", nvrtcCompileProgram) ||
        !load_symbol(nvrtc_module, "nvrtcGetProgramLogSize", nvrtcGetProgramLogSize) ||
        !load_symbol(nvrtc_module, "nvrtcGetProgramLog", nvrtcGetProgramLog) ||
        !load_symbol(nvrtc_module, "nvrtcGetPTXSize", nvrtcGetPtxSize) ||
        !load_symbol(nvrtc_module, "nvrtcGetPTX", nvrtcGetPtx) ||
        !load_symbol(nvrtc_module, "nvrtcDestroyProgram", nvrtcDestroyProgram) ||
        !load_symbol(cublas_module, "cublasCreate_v2", cublasCreate) ||
        !load_symbol(cublas_module, "cublasDestroy_v2", cublasDestroy) ||
        !load_symbol(cublas_module, "cublasSetStream_v2", cublasSetStream) ||
        !load_symbol(cublas_module, "cublasSgemm_v2", cublasSgemm)) return false;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        last_error = "cuBLAS handle creation failed";
        return false;
    }
    if (!compile_kernels()) return false;
    initialized = true;
    return true;
}

int launch_im2col(CUstream stream, CUdeviceptr x, CUdeviceptr cols,
                  int N, int CI, int H, int W, int KH, int KW, int OH, int OW,
                  int SH, int SW, int PH, int PW) {
    int total = N * OH * OW * CI * KH * KW;
    void* args[] = {&x, &cols, &N, &CI, &H, &W, &KH, &KW, &OH, &OW, &SH, &SW, &PH, &PW};
    return cuLaunchKernel(im2col_function, (total + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int launch_col2im(CUstream stream, CUdeviceptr grad_cols, CUdeviceptr dx,
                  int N, int CI, int H, int W, int KH, int KW, int OH, int OW,
                  int SH, int SW, int PH, int PW) {
    int total = N * CI * H * W;
    void* args[] = {&grad_cols, &dx, &N, &CI, &H, &W, &KH, &KW, &OH, &OW, &SH, &SW, &PH, &PW};
    return cuLaunchKernel(col2im_function, (total + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int launch_bias(CUstream stream, CUdeviceptr rows, CUdeviceptr bias, int M, int N) {
    int total = M * N;
    void* args[] = {&rows, &bias, &M, &N};
    return cuLaunchKernel(add_bias_function, (total + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int launch_bias_grad(CUstream stream, CUdeviceptr dy, CUdeviceptr db, int M, int CO) {
    void* args[] = {&dy, &db, &M, &CO};
    return cuLaunchKernel(bias_grad_function, (CO + 255) / 256, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int gemm(CUstream stream, int trans_a, int trans_b, int m, int n, int k,
         CUdeviceptr a, int lda, CUdeviceptr b, int ldb, CUdeviceptr c, int ldc) {
    if (cublasSetStream(cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) return 20;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemm(cublas_handle, trans_a, trans_b, m, n, k, &alpha,
                       reinterpret_cast<const float*>(a), lda,
                       reinterpret_cast<const float*>(b), ldb, &beta,
                       reinterpret_cast<float*>(c), ldc) == CUBLAS_STATUS_SUCCESS ? 0 : 21;
}

}  // namespace

extern "C" __declspec(dllexport) const char* mf_last_error() {
    return last_error.c_str();
}

extern "C" __declspec(dllexport) int mf_init(const char* cublas_path, const char* nvrtc_path) {
    return ensure_initialized(cublas_path, nvrtc_path) ? 0 : 1;
}

extern "C" __declspec(dllexport) int mf_forward(
    std::uint64_t x, std::uint64_t weight, std::uint64_t bias, std::uint64_t cols,
    std::uint64_t rows, int N, int CI, int H, int W, int CO, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW, std::uint64_t stream_ptr) {
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    int rc = launch_im2col(stream, x, cols, N, CI, H, W, KH, KW, OH, OW, SH, SW, PH, PW);
    if (rc != CUDA_SUCCESS) return 10;
    int M = N * OH * OW;
    int K = CI * KH * KW;
    rc = gemm(stream, CUBLAS_OP_T, CUBLAS_OP_N, CO, M, K, weight, K, cols, K, rows, CO);
    if (rc != 0) return rc;
    if (bias != 0) {
        rc = launch_bias(stream, rows, bias, M, CO);
        if (rc != CUDA_SUCCESS) return 11;
    }
    return 0;
}

extern "C" __declspec(dllexport) int mf_backward(
    std::uint64_t x, std::uint64_t weight, std::uint64_t dy, std::uint64_t cols,
    std::uint64_t dx, std::uint64_t dw, std::uint64_t grad_cols, std::uint64_t db,
    int N, int CI, int H, int W, int CO, int KH, int KW, int OH, int OW,
    int SH, int SW, int PH, int PW, std::uint64_t stream_ptr) {
    (void)x;
    CUstream stream = reinterpret_cast<CUstream>(stream_ptr);
    int M = N * OH * OW;
    int K = CI * KH * KW;
    // C = dy^T @ cols in row-major storage. The column-major cuBLAS view
    // therefore transposes the dy operand while keeping cols untransposed.
    int rc = gemm(stream, CUBLAS_OP_N, CUBLAS_OP_T, K, CO, M, cols, K, dy, CO, dw, K);
    if (rc != 0) return rc;
    rc = gemm(stream, CUBLAS_OP_N, CUBLAS_OP_N, K, M, CO, weight, K, dy, CO, grad_cols, K);
    if (rc != 0) return rc;
    rc = launch_col2im(stream, grad_cols, dx, N, CI, H, W, KH, KW, OH, OW, SH, SW, PH, PW);
    if (rc != CUDA_SUCCESS) return 12;
    if (db != 0) {
        rc = launch_bias_grad(stream, dy, db, M, CO);
        if (rc != CUDA_SUCCESS) return 13;
    }
    return 0;
}
