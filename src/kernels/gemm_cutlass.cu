// Layer 2 — Primitive kernels: CUTLASS GEMM backend.
//
// This file wires the fp32 GEMM (C = alpha*A@B + beta*C) onto the CUTLASS
// device API. The v1 bring-up uses cutlass::gemm::device::Gemm with the
// library's default kernel for the compiled arch (the same path proven by
// tests/smoke/cutlass_gemm_smoke.cu on sm_100).
//
// Layout mapping (ADR 0007): the caller (src/ops/linear.cc) does NOT pass plain
// row-major matrices — it builds transposed-stride views to express W^T without
// copying. So the GEMM must honor arbitrary {stride_row, stride_col} on each
// operand. Our Tensor2D places element (i,j) at i*stride_row + j*stride_col, so
// a 2-D view is either:
//   - row-major    when stride_col == 1  (leading dim = stride_row), or
//   - column-major when stride_row == 1  (leading dim = stride_col).
// CUTLASS device::Gemm is templated on the layout of each operand, so we detect
// each operand's orientation and dispatch to the matching instantiation. C here
// is always the row-major output buffer.

#include "src/kernels/gemm.h"

#include <string>

#include <cuda_bf16.h>

#include "cutlass/bfloat16.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "src/kernels/gemm_cublaslt_internal.h"

namespace gpt {
namespace kernels {

namespace {

// How a matrix's (row_stride, col_stride) map onto a CUTLASS matrix layout.
enum class Orientation { kRowMajor, kColumnMajor, kUnsupported };

// A GEMM operand contributes exactly one contiguous (unit-stride) axis. Detect
// which, and report the leading dimension (the stride of the *other* axis).
Orientation classify_strides(int64_t row_stride, int64_t col_stride,
                             int64_t* leading_dim) {
  if (col_stride == 1) {  // row-major: rows step by row_stride
    *leading_dim = row_stride;
    return Orientation::kRowMajor;
  }
  if (row_stride == 1) {  // column-major: cols step by col_stride
    *leading_dim = col_stride;
    return Orientation::kColumnMajor;
  }
  return Orientation::kUnsupported;
}

Orientation classify(const Tensor2D<float>& m, int64_t* leading_dim) {
  return classify_strides(m.stride[0], m.stride[1], leading_dim);
}

Orientation classify(const Tensor2D<__nv_bfloat16>& m, int64_t* leading_dim) {
  return classify_strides(m.stride[0], m.stride[1], leading_dim);
}

// Launch one concrete CUTLASS device::Gemm instantiation for the detected
// layouts. M,N,K and the leading dims are computed by the caller.
template <typename LayoutA, typename LayoutB>
cutlass::Status launch_gemm(int m, int n, int k, float alpha, const float* A,
                            int lda, const float* B, int ldb, float beta,
                            float* C, int ldc, cudaStream_t stream) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, LayoutA, float, LayoutB,
                                           float, ColumnMajor>;
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {A, lda}, {B, ldb}, {C, ldc},
                                {C, ldc}, {alpha, beta});
  return op(args, nullptr, stream);
}

template <typename LayoutA, typename LayoutB>
cutlass::Status launch_gemm_bf16(int m, int n, int k, float alpha,
                                 const __nv_bfloat16* A, int lda,
                                 const __nv_bfloat16* B, int ldb, float beta,
                                 __nv_bfloat16* C, int ldc,
                                 cudaStream_t stream) {
  using BFloat16 = cutlass::bfloat16_t;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::Gemm<BFloat16, LayoutA, BFloat16,
                                           LayoutB, BFloat16, ColumnMajor,
                                           float>;
  auto const* cutlass_a = reinterpret_cast<BFloat16 const*>(A);
  auto const* cutlass_b = reinterpret_cast<BFloat16 const*>(B);
  auto* cutlass_c = reinterpret_cast<BFloat16*>(C);
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {cutlass_a, lda}, {cutlass_b, ldb},
                                {cutlass_c, ldc}, {cutlass_c, ldc},
                                {alpha, beta});
  return op(args, nullptr, stream);
}

// Strided-batched analogue of launch_gemm: each batch shares the same layout
// and leading dim, and advances by a per-operand batch stride.
template <typename LayoutA, typename LayoutB>
cutlass::Status launch_gemm_batched(int m, int n, int k, float alpha,
                                    const float* A, int lda, int64_t batch_a,
                                    const float* B, int ldb, int64_t batch_b,
                                    float beta, float* C, int ldc,
                                    int64_t batch_c, int batch_count,
                                    cudaStream_t stream) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::GemmBatched<float, LayoutA, float,
                                                  LayoutB, float, ColumnMajor>;
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {A, lda}, batch_a, {B, ldb}, batch_b,
                                {C, ldc}, batch_c, {C, ldc}, batch_c,
                                {alpha, beta}, batch_count);
  return op(args, nullptr, stream);
}

template <typename LayoutA, typename LayoutB>
cutlass::Status launch_gemm_batched_bf16(
    int m, int n, int k, float alpha, const __nv_bfloat16* A, int lda,
    int64_t batch_a, const __nv_bfloat16* B, int ldb, int64_t batch_b,
    float beta, __nv_bfloat16* C, int ldc, int64_t batch_c, int batch_count,
    cudaStream_t stream) {
  using BFloat16 = cutlass::bfloat16_t;
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::GemmBatched<
      BFloat16, LayoutA, BFloat16, LayoutB, BFloat16, ColumnMajor, float>;
  auto const* cutlass_a = reinterpret_cast<BFloat16 const*>(A);
  auto const* cutlass_b = reinterpret_cast<BFloat16 const*>(B);
  auto* cutlass_c = reinterpret_cast<BFloat16*>(C);
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {cutlass_a, lda}, batch_a,
                                {cutlass_b, ldb}, batch_b, {cutlass_c, ldc},
                                batch_c, {cutlass_c, ldc}, batch_c,
                                {alpha, beta}, batch_count);
  return op(args, nullptr, stream);
}

}  // namespace

Status gemm_f32(Tensor2D<float> A, Tensor2D<float> B, Tensor2D<float> C,
                float alpha, float beta, const CudaStream& stream,
                GemmBackend backend) {
  if (backend == GemmBackend::kCublasLt) {
    return gemm_f32_cublaslt(A, B, C, alpha, beta, stream);
  }
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kInvalidArgument, "Unknown GEMM backend");
  }
  // Validate shapes.
  if (A.shape[1] != B.shape[0]) {
    return Status(StatusCode::kInvalidArgument, "GEMM K-dim mismatch");
  }
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    return Status(StatusCode::kInvalidArgument, "GEMM output shape mismatch");
  }

  const int M = static_cast<int>(A.shape[0]);
  const int N = static_cast<int>(B.shape[1]);
  const int K = static_cast<int>(A.shape[1]);

  // CUTLASS computes D = alpha*A@B + beta*C in column-major. We evaluate the
  // transpose instead: C^T = B^T @ A^T. Column-major C^T (N x M, ld=stride_row)
  // is bit-identical to our row-major C (M x N). The transpose of an operand
  // swaps its logical axes and its strides, which also flips row/column-major.
  // So passing B as the first factor and A as the second, each with row/col-
  // major SWAPPED relative to its own detection, yields C^T directly.
  int64_t lda64 = 0, ldb64 = 0, ldc64 = 0;
  const Orientation a_orient = classify(A, &lda64);
  const Orientation b_orient = classify(B, &ldb64);
  const Orientation c_orient = classify(C, &ldc64);
  if (a_orient == Orientation::kUnsupported ||
      b_orient == Orientation::kUnsupported ||
      c_orient != Orientation::kRowMajor) {
    return Status(StatusCode::kInvalidArgument,
                  "gemm_f32: unsupported strides (need unit-stride axis per "
                  "operand and a row-major output C)");
  }

  using RowMajor = cutlass::layout::RowMajor;
  using ColMajor = cutlass::layout::ColumnMajor;

  // For the transposed product C^T = B^T @ A^T, operand order is (B, A) and we
  // supply each operand's transposed layout: an operand that is row-major in
  // its own frame is column-major once transposed, and vice versa. The leading
  // dimension is unchanged by transposition (it is the physical stride).
  const int lda = static_cast<int>(ldb64);  // first factor is B
  const int ldb = static_cast<int>(lda64);  // second factor is A
  const int ldc = static_cast<int>(ldc64);
  const bool b_is_rm = (b_orient == Orientation::kRowMajor);
  const bool a_is_rm = (a_orient == Orientation::kRowMajor);

  cutlass::Status cs;
  // Dispatch over the transposed layouts of (B, A).
  if (!b_is_rm && !a_is_rm) {
    // B col-major -> B^T row-major; A col-major -> A^T row-major.
    cs = launch_gemm<RowMajor, RowMajor>(N, M, K, alpha, B.data, lda, A.data,
                                         ldb, beta, C.data, ldc, stream.get());
  } else if (!b_is_rm && a_is_rm) {
    cs = launch_gemm<RowMajor, ColMajor>(N, M, K, alpha, B.data, lda, A.data,
                                         ldb, beta, C.data, ldc, stream.get());
  } else if (b_is_rm && !a_is_rm) {
    cs = launch_gemm<ColMajor, RowMajor>(N, M, K, alpha, B.data, lda, A.data,
                                         ldb, beta, C.data, ldc, stream.get());
  } else {
    cs = launch_gemm<ColMajor, ColMajor>(N, M, K, alpha, B.data, lda, A.data,
                                         ldb, beta, C.data, ldc, stream.get());
  }

  if (cs != cutlass::Status::kSuccess) {
    return Status(StatusCode::kCudaError,
                  std::string("CUTLASS gemm_f32 launch failed: ") +
                      cutlassGetStatusString(cs));
  }
  // Surface any asynchronous launch error loudly rather than silently.
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

Status gemm_bf16(Tensor2D<__nv_bfloat16> A, Tensor2D<__nv_bfloat16> B,
                 Tensor2D<__nv_bfloat16> C, float alpha, float beta,
                 const CudaStream& stream, GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  if (A.shape[1] != B.shape[0]) {
    return Status(StatusCode::kInvalidArgument, "GEMM K-dim mismatch");
  }
  if (A.shape[0] != C.shape[0] || B.shape[1] != C.shape[1]) {
    return Status(StatusCode::kInvalidArgument, "GEMM output shape mismatch");
  }

  const int M = static_cast<int>(A.shape[0]);
  const int N = static_cast<int>(B.shape[1]);
  const int K = static_cast<int>(A.shape[1]);

  int64_t lda64 = 0, ldb64 = 0, ldc64 = 0;
  const Orientation a_orient = classify(A, &lda64);
  const Orientation b_orient = classify(B, &ldb64);
  const Orientation c_orient = classify(C, &ldc64);
  if (a_orient == Orientation::kUnsupported ||
      b_orient == Orientation::kUnsupported ||
      c_orient != Orientation::kRowMajor) {
    return Status(StatusCode::kInvalidArgument,
                  "gemm_bf16: unsupported strides (need unit-stride axis per "
                  "operand and a row-major output C)");
  }

  using RowMajor = cutlass::layout::RowMajor;
  using ColMajor = cutlass::layout::ColumnMajor;

  const int lda = static_cast<int>(ldb64);
  const int ldb = static_cast<int>(lda64);
  const int ldc = static_cast<int>(ldc64);
  const bool b_is_rm = (b_orient == Orientation::kRowMajor);
  const bool a_is_rm = (a_orient == Orientation::kRowMajor);

  cutlass::Status cs;
  if (!b_is_rm && !a_is_rm) {
    cs = launch_gemm_bf16<RowMajor, RowMajor>(
        N, M, K, alpha, B.data, lda, A.data, ldb, beta, C.data, ldc,
        stream.get());
  } else if (!b_is_rm && a_is_rm) {
    cs = launch_gemm_bf16<RowMajor, ColMajor>(
        N, M, K, alpha, B.data, lda, A.data, ldb, beta, C.data, ldc,
        stream.get());
  } else if (b_is_rm && !a_is_rm) {
    cs = launch_gemm_bf16<ColMajor, RowMajor>(
        N, M, K, alpha, B.data, lda, A.data, ldb, beta, C.data, ldc,
        stream.get());
  } else {
    cs = launch_gemm_bf16<ColMajor, ColMajor>(
        N, M, K, alpha, B.data, lda, A.data, ldb, beta, C.data, ldc,
        stream.get());
  }

  if (cs != cutlass::Status::kSuccess) {
    return Status(StatusCode::kCudaError,
                  std::string("CUTLASS gemm_bf16 launch failed: ") +
                      cutlassGetStatusString(cs));
  }
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

Status batched_gemm_f32(Tensor3D<float> A, Tensor3D<float> B,
                        Tensor3D<float> C, float alpha, float beta,
                        const CudaStream& stream, GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  if (A.shape[2] != B.shape[1]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM K-dim mismatch");
  }
  if (A.shape[1] != C.shape[1] || B.shape[2] != C.shape[2]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM output shape mismatch");
  }

  const int batch_count = static_cast<int>(A.shape[0]);
  const int M = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[2]);
  const int K = static_cast<int>(A.shape[2]);

  // Same transpose trick as gemm_f32, applied per batch. Each operand's 2-D
  // sub-matrix uses strides {stride[1], stride[2]}; the batch stride is
  // stride[0]. Output C is row-major (stride[2]==1) so column-major C^T is
  // bit-identical, and we compute C^T = B^T @ A^T with operand order (B, A).
  int64_t lda64 = 0, ldb64 = 0, ldc64 = 0;
  const Orientation a_orient = classify_strides(A.stride[1], A.stride[2], &lda64);
  const Orientation b_orient = classify_strides(B.stride[1], B.stride[2], &ldb64);
  const Orientation c_orient = classify_strides(C.stride[1], C.stride[2], &ldc64);
  if (a_orient == Orientation::kUnsupported ||
      b_orient == Orientation::kUnsupported ||
      c_orient != Orientation::kRowMajor) {
    return Status(StatusCode::kInvalidArgument,
                  "batched_gemm_f32: unsupported strides (need unit-stride axis "
                  "per operand and a row-major output C)");
  }

  using RowMajor = cutlass::layout::RowMajor;
  using ColMajor = cutlass::layout::ColumnMajor;

  const int lda = static_cast<int>(ldb64);  // first factor is B
  const int ldb = static_cast<int>(lda64);  // second factor is A
  const int ldc = static_cast<int>(ldc64);
  const int64_t batch_a = B.stride[0];  // first factor is B
  const int64_t batch_b = A.stride[0];  // second factor is A
  const int64_t batch_c = C.stride[0];
  const bool b_is_rm = (b_orient == Orientation::kRowMajor);
  const bool a_is_rm = (a_orient == Orientation::kRowMajor);

  cutlass::Status cs;
  if (!b_is_rm && !a_is_rm) {
    cs = launch_gemm_batched<RowMajor, RowMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  } else if (!b_is_rm && a_is_rm) {
    cs = launch_gemm_batched<RowMajor, ColMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  } else if (b_is_rm && !a_is_rm) {
    cs = launch_gemm_batched<ColMajor, RowMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  } else {
    cs = launch_gemm_batched<ColMajor, ColMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  }

  if (cs != cutlass::Status::kSuccess) {
    return Status(StatusCode::kCudaError,
                  std::string("CUTLASS batched_gemm_f32 launch failed: ") +
                      cutlassGetStatusString(cs));
  }
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

Status batched_gemm_bf16(Tensor3D<__nv_bfloat16> A,
                         Tensor3D<__nv_bfloat16> B,
                         Tensor3D<__nv_bfloat16> C, float alpha, float beta,
                         const CudaStream& stream, GemmBackend backend) {
  if (backend != GemmBackend::kCutlass) {
    return Status(StatusCode::kNotImplemented,
                  "Only CUTLASS backend implemented here");
  }
  if (A.shape[0] != B.shape[0] || A.shape[0] != C.shape[0]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM batch-dim mismatch");
  }
  if (A.shape[2] != B.shape[1]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM K-dim mismatch");
  }
  if (A.shape[1] != C.shape[1] || B.shape[2] != C.shape[2]) {
    return Status(StatusCode::kInvalidArgument,
                  "Batched GEMM output shape mismatch");
  }

  const int batch_count = static_cast<int>(A.shape[0]);
  const int M = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[2]);
  const int K = static_cast<int>(A.shape[2]);

  int64_t lda64 = 0, ldb64 = 0, ldc64 = 0;
  const Orientation a_orient = classify_strides(A.stride[1], A.stride[2], &lda64);
  const Orientation b_orient = classify_strides(B.stride[1], B.stride[2], &ldb64);
  const Orientation c_orient = classify_strides(C.stride[1], C.stride[2], &ldc64);
  if (a_orient == Orientation::kUnsupported ||
      b_orient == Orientation::kUnsupported ||
      c_orient != Orientation::kRowMajor) {
    return Status(StatusCode::kInvalidArgument,
                  "batched_gemm_bf16: unsupported strides (need unit-stride axis "
                  "per operand and a row-major output C)");
  }

  using RowMajor = cutlass::layout::RowMajor;
  using ColMajor = cutlass::layout::ColumnMajor;

  const int lda = static_cast<int>(ldb64);
  const int ldb = static_cast<int>(lda64);
  const int ldc = static_cast<int>(ldc64);
  const int64_t batch_a = B.stride[0];
  const int64_t batch_b = A.stride[0];
  const int64_t batch_c = C.stride[0];
  const bool b_is_rm = (b_orient == Orientation::kRowMajor);
  const bool a_is_rm = (a_orient == Orientation::kRowMajor);

  cutlass::Status cs;
  if (!b_is_rm && !a_is_rm) {
    cs = launch_gemm_batched_bf16<RowMajor, RowMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  } else if (!b_is_rm && a_is_rm) {
    cs = launch_gemm_batched_bf16<RowMajor, ColMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  } else if (b_is_rm && !a_is_rm) {
    cs = launch_gemm_batched_bf16<ColMajor, RowMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  } else {
    cs = launch_gemm_batched_bf16<ColMajor, ColMajor>(
        N, M, K, alpha, B.data, lda, batch_a, A.data, ldb, batch_b, beta,
        C.data, ldc, batch_c, batch_count, stream.get());
  }

  if (cs != cutlass::Status::kSuccess) {
    return Status(StatusCode::kCudaError,
                  std::string("CUTLASS batched_gemm_bf16 launch failed: ") +
                      cutlassGetStatusString(cs));
  }
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

}  // namespace kernels
}  // namespace gpt
