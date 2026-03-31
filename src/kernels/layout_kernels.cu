// Layer 2 — Primitive kernels: layout transform CUDA implementations.

#include "src/kernels/layout_kernels.h"

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>

namespace gpt {
namespace kernels {

namespace {

static constexpr int kTileSize = 32;

// Tiled 2D transpose using shared memory.
__global__ void transpose_2d_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t M, int64_t N) {
  __shared__ float tile[kTileSize][kTileSize + 1];  // +1 avoids bank conflicts

  int64_t bx = blockIdx.x * kTileSize;
  int64_t by = blockIdx.y * kTileSize;
  int64_t ix = bx + threadIdx.x;
  int64_t iy = by + threadIdx.y;

  if (ix < N && iy < M) {
    tile[threadIdx.y][threadIdx.x] = input[iy * N + ix];
  }
  __syncthreads();

  // Transposed write coordinates.
  int64_t ox = by + threadIdx.x;
  int64_t oy = bx + threadIdx.y;
  if (ox < M && oy < N) {
    output[oy * M + ox] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void split_heads_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t B, int64_t T, int64_t H,
                                   int64_t Dh) {
  // input layout:  [B, T, H*Dh]  row-major
  // output layout: [B, H, T, Dh] row-major
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * H * T * Dh;
  if (idx >= total) return;

  int64_t d = idx % Dh;
  int64_t t = (idx / Dh) % T;
  int64_t h = (idx / (Dh * T)) % H;
  int64_t b = idx / (Dh * T * H);

  int64_t in_idx = b * (T * H * Dh) + t * (H * Dh) + h * Dh + d;
  output[idx] = input[in_idx];
}

__global__ void merge_heads_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t B, int64_t H, int64_t T,
                                   int64_t Dh) {
  // input layout:  [B, H, T, Dh] row-major
  // output layout: [B, T, H*Dh]  row-major
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t D = H * Dh;
  int64_t total = B * T * D;
  if (idx >= total) return;

  int64_t d = idx % D;
  int64_t t = (idx / D) % T;
  int64_t b = idx / (D * T);

  int64_t h = d / Dh;
  int64_t dh = d % Dh;

  int64_t in_idx = b * (H * T * Dh) + h * (T * Dh) + t * Dh + dh;
  output[idx] = input[in_idx];
}

// Fused QKV split: read from packed [B, T, 3*D] and write Q/K/V as [B, H, T, Dh].
// Each thread handles one element across all three outputs.
__global__ void split_qkv_heads_kernel(const float* __restrict__ qkv,
                                       float* __restrict__ Q,
                                       float* __restrict__ K,
                                       float* __restrict__ V,
                                       int64_t B, int64_t T, int64_t H,
                                       int64_t Dh) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * H * T * Dh;
  if (idx >= total) return;

  // Decompose output index [B, H, T, Dh] row-major.
  int64_t d = idx % Dh;
  int64_t t = (idx / Dh) % T;
  int64_t h = (idx / (Dh * T)) % H;
  int64_t b = idx / (Dh * T * H);

  int64_t D = H * Dh;
  // QKV is packed as [B, T, 3*D] with Q at col 0..D-1, K at D..2D-1, V at 2D..3D-1.
  int64_t base = b * (T * 3 * D) + t * (3 * D) + h * Dh + d;
  Q[idx] = qkv[base];
  K[idx] = qkv[base + D];
  V[idx] = qkv[base + 2 * D];
}

// Generic strided 2D copy: handles arbitrary row strides for src and dst.
__global__ void strided_copy_2d_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       int64_t rows, int64_t cols,
                                       int64_t src_row_stride,
                                       int64_t dst_row_stride) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * cols;
  if (idx >= total) return;

  int64_t col = idx % cols;
  int64_t row = idx / cols;
  dst[row * dst_row_stride + col] = src[row * src_row_stride + col];
}

// Compute 1D grid size, clamping to CUDA's max grid.x limit.
inline int grid_1d(int64_t total, int block_size) {
  int64_t g = (total + block_size - 1) / block_size;
  return static_cast<int>(std::min(g, static_cast<int64_t>(INT_MAX)));
}

}  // namespace

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

Status transpose_2d_f32(Tensor2D<const float> input,
                        Tensor2D<float> output,
                        const CudaStream& stream) {
  int64_t M = input.shape[0];
  int64_t N = input.shape[1];
  if (output.shape[0] != N || output.shape[1] != M) {
    return Status(StatusCode::kInvalidArgument,
                  "transpose_2d output shape mismatch");
  }
  dim3 block(kTileSize, kTileSize);
  dim3 grid(static_cast<unsigned>((N + kTileSize - 1) / kTileSize),
            static_cast<unsigned>((M + kTileSize - 1) / kTileSize));
  transpose_2d_kernel<<<grid, block, 0, stream.get()>>>(
      input.data, output.data, M, N);
  return Status::Ok();
}

Status repack_contiguous_f32(Tensor2D<const float> input,
                             Tensor2D<float> output,
                             const CudaStream& stream) {
  if (input.shape != output.shape) {
    return Status(StatusCode::kInvalidArgument,
                  "repack_contiguous shape mismatch");
  }
  // If input is already contiguous, just memcpy.
  if (input.is_contiguous() && output.is_contiguous()) {
    size_t bytes = static_cast<size_t>(input.numel()) * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(output.data, input.data, bytes,
                                       cudaMemcpyDeviceToDevice, stream.get());
    if (err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(err));
    }
    return Status::Ok();
  }
  // Strided copy for non-contiguous inputs/outputs.
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  int64_t total = rows * cols;
  int block = 256;
  int grid = grid_1d(total, block);
  strided_copy_2d_kernel<<<grid, block, 0, stream.get()>>>(
      input.data, output.data, rows, cols,
      input.stride[0], output.stride[0]);
  return Status::Ok();
}

Status split_heads_f32(Tensor3D<const float> input,
                       Tensor4D<float> output,
                       int64_t n_heads,
                       const CudaStream& stream) {
  int64_t B = input.shape[0];
  int64_t T = input.shape[1];
  int64_t D = input.shape[2];
  int64_t Dh = D / n_heads;
  if (D % n_heads != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "D not divisible by n_heads in split_heads");
  }
  int64_t total = B * n_heads * T * Dh;
  int block = 256;
  int grid = grid_1d(total, block);
  split_heads_kernel<<<grid, block, 0, stream.get()>>>(
      input.data, output.data, B, T, n_heads, Dh);
  return Status::Ok();
}

Status merge_heads_f32(Tensor4D<const float> input,
                       Tensor3D<float> output,
                       const CudaStream& stream) {
  int64_t B = input.shape[0];
  int64_t H = input.shape[1];
  int64_t T = input.shape[2];
  int64_t Dh = input.shape[3];
  int64_t D = H * Dh;
  int64_t total = B * T * D;
  int block = 256;
  int grid = grid_1d(total, block);
  merge_heads_kernel<<<grid, block, 0, stream.get()>>>(
      input.data, output.data, B, H, T, Dh);
  return Status::Ok();
}

Status split_qkv_heads_f32(Tensor3D<const float> qkv,
                           Tensor4D<float> Q,
                           Tensor4D<float> K,
                           Tensor4D<float> V,
                           int64_t n_heads,
                           const CudaStream& stream) {
  int64_t B = qkv.shape[0];
  int64_t T = qkv.shape[1];
  int64_t three_D = qkv.shape[2];
  if (three_D % 3 != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "split_qkv_heads: last dim must be 3*D");
  }
  int64_t D = three_D / 3;
  int64_t Dh = D / n_heads;
  if (D % n_heads != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "D not divisible by n_heads in split_qkv_heads");
  }
  int64_t total = B * n_heads * T * Dh;
  int block = 256;
  int grid = grid_1d(total, block);
  split_qkv_heads_kernel<<<grid, block, 0, stream.get()>>>(
      qkv.data, Q.data, K.data, V.data, B, T, n_heads, Dh);
  return Status::Ok();
}

}  // namespace kernels
}  // namespace gpt
