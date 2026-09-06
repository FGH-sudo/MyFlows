// FP32 NCHW im2col/col2im kernels for the stage-two controlled comparison.
// One thread writes one output element; the matrix multiplication remains the
// same CuPy GEMM used by the reference im2col path.
extern "C" __global__ void conv2d_im2col_forward(
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

extern "C" __global__ void conv2d_col2im_backward(
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
