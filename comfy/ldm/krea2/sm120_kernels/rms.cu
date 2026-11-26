#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ float warp_sum(float x) {
  for (int offset = 16; offset; offset >>= 1)
    x += __shfl_down_sync(0xffffffff, x, offset);
  return x;
}
template <int D>
__global__ void rms(const half *x, const float *weight, half *out, int h, int l,
                    int sh, int sl, float eps) {
  constexpr int threads = D == 128 ? 32 : 128;
  constexpr int groups = D / (threads * 4);
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int row = D == 128 ? blockIdx.x * 4 + warp : blockIdx.x;
  if (row >= h * l)
    return;
  int ti = D == 128 ? lane : threadIdx.x;
  int offset = (row / l) * sh + (row % l) * sl;
  const uint2 *src = reinterpret_cast<const uint2 *>(x + offset);
  uint2 packed[groups];
  float s = 0;
#pragma unroll
  for (int j = 0; j < groups; j++) {
    packed[j] = src[ti + j * threads];
    half *values = reinterpret_cast<half *>(&packed[j]);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float v = __half2float(values[k]);
      s = fmaf(v, v, s);
    }
  }
  s = warp_sum(s);
  float inv;
  if constexpr (D == 128) {
    inv = rsqrtf(__shfl_sync(0xffffffff, s, 0) / 128.f + eps);
  } else {
    __shared__ float a[4];
    if (!lane)
      a[warp] = s;
    __syncthreads();
    if (!threadIdx.x)
      a[0] = rsqrtf(__fdiv_rn((a[0] + a[2]) + (a[1] + a[3]), float(D)) + eps);
    __syncthreads();
    inv = a[0];
  }
  uint2 *dst = reinterpret_cast<uint2 *>(out + row * D);
#pragma unroll
  for (int j = 0; j < groups; j++) {
    int c = (ti + j * threads) * 4;
    half *v = reinterpret_cast<half *>(&packed[j]);
#pragma unroll
    for (int k = 0; k < 4; k++)
      v[k] = __float2half_rn(weight[c + k] * (inv * __half2float(v[k])));
    dst[c / 4] = packed[j];
  }
}
extern "C" __declspec(dllexport) int launch_rms(const half *x, const float *w,
                                                half *y, int h, int l, int d,
                                                int sh, int sl, float eps,
                                                cudaStream_t stream) {
  if (d == 128)
    rms<128><<<(h * l + 3) / 4, 128, 0, stream>>>(x, w, y, h, l, sh, sl, eps);
  else if (d == 6144)
    rms<6144><<<h * l, 128, 0, stream>>>(x, w, y, h, l, sh, sl, eps);
  else
    return -1;
  return cudaGetLastError();
}
