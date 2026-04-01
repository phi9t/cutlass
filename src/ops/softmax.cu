// Layer 3 — Composite neural ops: masked softmax CUDA implementation.

#include "src/ops/softmax.h"

#include <cfloat>
#include <climits>
#include <cuda_runtime.h>

namespace gpt {
namespace ops {

namespace {

// One block per row.  V1 reference: single-thread-per-row for correctness.
__global__ void masked_softmax_forward_kernel(
    const float* __restrict__ input,
    const int8_t* __restrict__ mask,  // may be null
    float* __restrict__ output,
    int64_t rows, int64_t cols) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;

  const float* in_row = input + row * cols;
  float* out_row = output + row * cols;
  const int8_t* mask_row = mask ? mask + row * cols : nullptr;

  // Step 1: row max over valid positions.
  float row_max_val = -FLT_MAX;
  for (int64_t j = 0; j < cols; ++j) {
    bool masked = mask_row && mask_row[j];
    if (!masked) {
      row_max_val = fmaxf(row_max_val, in_row[j]);
    }
  }

  // Step 2-3: exp(x - max) on valid positions.
  float row_sum_val = 0.0f;
  for (int64_t j = 0; j < cols; ++j) {
    bool masked = mask_row && mask_row[j];
    if (!masked) {
      float val = expf(in_row[j] - row_max_val);
      out_row[j] = val;
      row_sum_val += val;
    } else {
      out_row[j] = 0.0f;
    }
  }

  // Step 4-5: normalize.
  if (row_sum_val > 0.0f) {
    float inv_sum = 1.0f / row_sum_val;
    for (int64_t j = 0; j < cols; ++j) {
      out_row[j] *= inv_sum;
    }
  }
}

// Backward: dX = P * (dP - dot(P, dP))
__global__ void masked_softmax_backward_kernel(
    const float* __restrict__ dP,
    const float* __restrict__ P,
    float* __restrict__ dX,
    int64_t rows, int64_t cols) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;

  const float* dP_row = dP + row * cols;
  const float* P_row = P + row * cols;
  float* dX_row = dX + row * cols;

  // dot(P, dP) for this row.
  float dot = 0.0f;
  for (int64_t j = 0; j < cols; ++j) {
    dot += P_row[j] * dP_row[j];
  }

  // dX[j] = P[j] * (dP[j] - dot)
  for (int64_t j = 0; j < cols; ++j) {
    dX_row[j] = P_row[j] * (dP_row[j] - dot);
  }
}

}  // namespace

Status masked_softmax_forward(Tensor2D<const float> input,
                              Tensor2D<const int8_t> mask,
                              Tensor2D<float> output,
                              const CudaStream& stream) {
  int64_t rows = input.shape[0];
  int64_t cols = input.shape[1];
  if (output.shape[0] != rows || output.shape[1] != cols) {
    return Status(StatusCode::kInvalidArgument,
                  "masked_softmax_forward: output shape mismatch");
  }

  if (rows > INT_MAX) {
    return Status(StatusCode::kInvalidArgument,
                  "masked_softmax_forward: rows exceeds CUDA grid limit");
  }
  // V1 reference: one block per row, one thread.
  masked_softmax_forward_kernel<<<static_cast<int>(rows), 1, 0, stream.get()>>>(
      input.data, mask.data, output.data, rows, cols);
  return Status::Ok();
}

Status masked_softmax_backward(Tensor2D<const float> dP,
                               Tensor2D<const float> P,
                               Tensor2D<float> dX,
                               const CudaStream& stream) {
  int64_t rows = dP.shape[0];
  int64_t cols = dP.shape[1];

  if (rows > INT_MAX) {
    return Status(StatusCode::kInvalidArgument,
                  "masked_softmax_backward: rows exceeds CUDA grid limit");
  }
  masked_softmax_backward_kernel<<<static_cast<int>(rows), 1, 0, stream.get()>>>(
      dP.data, P.data, dX.data, rows, cols);
  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
