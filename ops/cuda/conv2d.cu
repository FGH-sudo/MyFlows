extern "C" __global__ void conv2d_forward_direct(
    const float* x, const float* w, const float* b, float* y,
    int N, int CI, int H, int W, int CO, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW, int has_bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * CO * OH * OW) return;
    int ow = idx % OW, oh = (idx / OW) % OH;
    int oc = (idx / (OW * OH)) % CO, n = idx / (OW * OH * CO);
    float sum = 0.0f;
    for (int ic = 0; ic < CI; ++ic)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {
                long long ih = (long long)oh * SH - PH + kh;
                long long iw = (long long)ow * SW - PW + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    sum += x[((n * CI + ic) * H + ih) * W + iw]
                         * w[((oc * CI + ic) * KH + kh) * KW + kw];
            }
    y[idx] = sum + (has_bias ? b[oc] : 0.0f);
}

extern "C" __global__ void conv2d_backward_input(
    const float* w, const float* dy, float* dx,
    int N, int CI, int H, int W, int CO, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * CI * H * W) return;
    int iw = idx % W, ih = (idx / W) % H;
    int ic = (idx / (W * H)) % CI, n = idx / (W * H * CI);
    // Enumerate only output windows touching this input. Descending output
    // coordinates preserve the baseline's ascending kh/kw accumulation order.
    int oh0 = (int)max(0LL, ((long long)ih + PH - KH + SH) / SH);
    int oh1 = (int)min((long long)OH - 1, ((long long)ih + PH) / SH);
    int ow0 = (int)max(0LL, ((long long)iw + PW - KW + SW) / SW);
    int ow1 = (int)min((long long)OW - 1, ((long long)iw + PW) / SW);
    float sum = 0.0f;
    for (int oc = 0; oc < CO; ++oc)
        for (int oh = oh1; oh >= oh0; --oh)
            for (int ow = ow1; ow >= ow0; --ow) {
                int kh = (int)((long long)ih + PH - (long long)oh * SH);
                int kw = (int)((long long)iw + PW - (long long)ow * SW);
                sum += dy[((n * CO + oc) * OH + oh) * OW + ow]
                     * w[((oc * CI + ic) * KH + kh) * KW + kw];
            }
    dx[idx] = sum;
}

extern "C" __global__ void conv2d_backward_weight(
    const float* x, const float* dy, float* dw,
    int N, int CI, int H, int W, int CO, int KH, int KW,
    int OH, int OW, int SH, int SW, int PH, int PW) {
    int idx = blockIdx.x;
    if (idx >= CO * CI * KH * KW) return;
    int kw = idx % KW, kh = (idx / KW) % KH;
    int ic = (idx / (KW * KH)) % CI, oc = idx / (KW * KH * CI);
    int tid = threadIdx.x;
    float sum = 0.0f;
    int positions = N * OH * OW;
    for (int flat = tid; flat < positions; flat += blockDim.x) {
        int pos = flat % (OH * OW);
        int n = flat / (OH * OW);
        int oh = pos / OW;
        int ow = pos % OW;
        int ih = oh * SH - PH + kh;
        int iw = ow * SW - PW + kw;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            sum += x[((n * CI + ic) * H + ih) * W + iw]
                 * dy[((n * CO + oc) * OH + oh) * OW + ow];
    }
    __shared__ float partial[512];
    partial[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) partial[tid] += partial[tid + offset];
        __syncthreads();
    }
    if (tid == 0) dw[idx] = partial[0];
}

extern "C" __global__ void conv2d_backward_bias(
    const float* dy, float* db, int N, int CO, int OH, int OW) {
    int oc = blockIdx.x;
    if (oc >= CO) return;
    int tid = threadIdx.x;
    float sum = 0.0f;
    int positions = N * OH * OW;
    for (int flat = tid; flat < positions; flat += blockDim.x) {
        int n = flat / (OH * OW);
        int pos = flat % (OH * OW);
        sum += dy[(n * CO + oc) * OH * OW + pos];
    }
    __shared__ float partial[512];
    partial[tid] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) partial[tid] += partial[tid + offset];
        __syncthreads();
    }
    if (tid == 0) db[oc] = partial[0];
}
