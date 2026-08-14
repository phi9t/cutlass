// Layer 2 — Primitive kernels: reduction implementations.

#include "src/kernels/reductions.h"

#include <cfloat>
#include <cuda_runtime.h>

namespace gpt {
namespace kernels {

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

namespace {

__global__ void row_max_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t rows, int64_t cols) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float val = -FLT_MAX;
  for (int64_t j = 0; j < cols; ++j) {
    val = fmaxf(val, row_ptr[j]);
  }
  output[row] = val;
}

__global__ void row_sum_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t rows, int64_t cols) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float val = 0.0f;
  for (int64_t j = 0; j < cols; ++j) {
    val += row_ptr[j];
  }
  output[row] = val;
}

__global__ void col_sum_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t rows, int64_t cols) {
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) return;

  float val = 0.0f;
  for (int64_t row = 0; row < rows; ++row) {
    val += input[row * cols + col];
  }
  output[col] = val;
}

__global__ void row_mean_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t rows, int64_t cols) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;
  float val = 0.0f;
  for (int64_t j = 0; j < cols; ++j) {
    val += row_ptr[j];
  }
  output[row] = val / static_cast<float>(cols);
}

__global__ void row_variance_kernel(const float* __restrict__ input,
                                    float* __restrict__ var_out,
                                    float* __restrict__ mean_out,
                                    int64_t rows, int64_t cols) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;

  // Two-pass for numerical stability.
  float mean = 0.0f;
  for (int64_t j = 0; j < cols; ++j) mean += row_ptr[j];
  mean /= static_cast<float>(cols);

  float var = 0.0f;
  for (int64_t j = 0; j < cols; ++j) {
    float diff = row_ptr[j] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(cols);

  if (mean_out) mean_out[row] = mean;
  var_out[row] = var;
}

__global__ void row_mean_inv_std_kernel(const float* __restrict__ input,
                                        float* __restrict__ mean_out,
                                        float* __restrict__ inv_std_out,
                                        int64_t rows, int64_t cols,
                                        float eps) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;

  const float* row_ptr = input + row * cols;

  float mean = 0.0f;
  for (int64_t j = 0; j < cols; ++j) mean += row_ptr[j];
  mean /= static_cast<float>(cols);

  float var = 0.0f;
  for (int64_t j = 0; j < cols; ++j) {
    float diff = row_ptr[j] - mean;
    var += diff * diff;
  }
  var /= static_cast<float>(cols);

  mean_out[row] = mean;
  inv_std_out[row] = rsqrtf(var + eps);
}

}  // namespace

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

static constexpr int kBlockSize = 256;

Status row_max_f32(Tensor2D<const float> input, Tensor1D<float> output,
                   const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (output.shape[0] != rows) {
    return Status(StatusCode::kInvalidArgument, "row_max output shape mismatch");
  }
  int grid = static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
  row_max_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, output.data, rows, cols);
  return Status::Ok();
}

Status row_sum_f32(Tensor2D<const float> input, Tensor1D<float> output,
                   const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (output.shape[0] != rows) {
    return Status(StatusCode::kInvalidArgument, "row_sum output shape mismatch");
  }
  int grid = static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
  row_sum_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, output.data, rows, cols);
  return Status::Ok();
}

Status col_sum_f32(Tensor2D<const float> input, Tensor1D<float> output,
                   const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (output.shape[0] != cols) {
    return Status(StatusCode::kInvalidArgument, "col_sum output shape mismatch");
  }
  int grid = static_cast<int>((cols + kBlockSize - 1) / kBlockSize);
  col_sum_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, output.data, rows, cols);
  return Status::Ok();
}

Status row_mean_f32(Tensor2D<const float> input, Tensor1D<float> output,
                    const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (output.shape[0] != rows) {
    return Status(StatusCode::kInvalidArgument,
                  "row_mean output shape mismatch");
  }
  int grid = static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
  row_mean_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, output.data, rows, cols);
  return Status::Ok();
}

Status row_variance_f32(Tensor2D<const float> input,
                        Tensor1D<float> variance,
                        Tensor1D<float> mean,
                        const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (variance.shape[0] != rows) {
    return Status(StatusCode::kInvalidArgument,
                  "row_variance output shape mismatch");
  }
  int grid = static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
  row_variance_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, variance.data, mean.data, rows, cols);
  return Status::Ok();
}

Status row_mean_inv_std_f32(Tensor2D<const float> input,
                            Tensor1D<float> mean,
                            Tensor1D<float> inv_std,
                            float eps,
                            const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (mean.shape[0] != rows || inv_std.shape[0] != rows) {
    return Status(StatusCode::kInvalidArgument,
                  "row_mean_inv_std output shape mismatch");
  }
  int grid = static_cast<int>((rows + kBlockSize - 1) / kBlockSize);
  row_mean_inv_std_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, mean.data, inv_std.data, rows, cols, eps);
  return Status::Ok();
}

}  // namespace kernels
}  // namespace gpt
