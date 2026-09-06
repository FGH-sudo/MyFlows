extern "C" __global__ void maxpool2d_forward_direct(
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

extern "C" __global__ void maxpool2d_backward_gather(
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
