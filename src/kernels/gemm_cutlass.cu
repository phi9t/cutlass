// Layer 2 — Primitive kernels: CUTLASS GEMM backend.
//
// Uses the CUTLASS 2.x device-level GEMM API (SM80 Ampere defaults).
// Row-major layouts with sensible tile shapes for training workloads.
// bf16 paths use fp32 accumulation for numerical stability.

#include "src/kernels/gemm.h"

#include <cuda_bf16.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace gpt {
namespace kernels {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

gpt::Status cutlass_to_gpt_status(cutlass::Status s) {
  switch (s) {
    case cutlass::Status::kSuccess:
      return gpt::Status::Ok();
    case cutlass::Status::kErrorMisalignedOperand:
      return gpt::Status(StatusCode::kInvalidArgument,
                         "CUTLASS: misaligned operand");
    case cutlass::Status::kErrorInvalidProblem:
      return gpt::Status(StatusCode::kInvalidArgument,
                         "CUTLASS: invalid problem size");
    default:
      return gpt::Status(StatusCode::kInternalError,
                         "CUTLASS kernel launch failed");
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Type aliases — CUTLASS GEMM configurations
// ---------------------------------------------------------------------------

// f32 GEMM: A[M,K] row-major x B[K,N] row-major -> C[M,N] row-major
// Uses SM80 Tensor-Core-accelerated SIMT path.
using GemmF32 = cutlass::gemm::device::Gemm<
    float,                                    // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    float,                                    // ElementB
    cutlass::layout::RowMajor,                // LayoutB
    float,                                    // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                    // ElementAccumulator
    cutlass::arch::OpClassSimt,               // OperatorClass
    cutlass::arch::Sm80                       // ArchTag
>;

// bf16 GEMM with fp32 accumulation: uses Tensor Core OpClassTensorOp on SM80.
using GemmBf16 = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t,                      // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::bfloat16_t,                      // ElementB
    cutlass::layout::RowMajor,                // LayoutB
    cutlass::bfloat16_t,                      // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                    // ElementAccumulator (fp32)
    cutlass::arch::OpClassTensorOp,           // OperatorClass
    cutlass::arch::Sm80                       // ArchTag
>;

// Batched f32 GEMM
using BatchedGemmF32 = cutlass::gemm::device::GemmBatched<
    float,                                    // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    float,                                    // ElementB
    cutlass::layout::RowMajor,                // LayoutB
    float,                                    // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                    // ElementAccumulator
    cutlass::arch::OpClassSimt,               // OperatorClass
    cutlass::arch::Sm80                       // ArchTag
>;

// Batched bf16 GEMM with fp32 accumulation
using BatchedGemmBf16 = cutlass::gemm::device::GemmBatched<
    cutlass::bfloat16_t,                      // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::bfloat16_t,                      // ElementB
    cutlass::layout::RowMajor,                // LayoutB
    cutlass::bfloat16_t,                      // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                    // ElementAccumulator (fp32)
    cutlass::arch::OpClassTensorOp,           // OperatorClass
    cutlass::arch::Sm80                       // ArchTag
>;

// ---------------------------------------------------------------------------
// GEMM implementations
// ---------------------------------------------------------------------------

Status gemm_f32(Tensor2D<float> A, Tensor2D<float> B, Tensor2D<float> C,
                float alpha, float beta, const CudaStream& stream,
                GemmBackend backend) {
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

  int M = static_cast<int>(A.shape[0]);
  int N = static_cast<int>(B.shape[1]);
  int K = static_cast<int>(A.shape[1]);

  int lda = static_cast<int>(A.stride[0]);  // row-major: stride of dim-0
  int ldb = static_cast<int>(B.stride[0]);
  int ldc = static_cast<int>(C.stride[0]);

  GemmF32 gemm_op;
  GemmF32::Arguments args(
      {M, N, K},                                              // problem_size
      {A.data, lda},                                          // ref_A
      {B.data, ldb},                                          // ref_B
      {C.data, ldc},                                          // ref_C (source)
      {C.data, ldc},                                          // ref_D (destination)
      {alpha, beta}                                           // epilogue params
  );

  cutlass::Status status = gemm_op(args, nullptr, stream.get());
  return cutlass_to_gpt_status(status);
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

  int M = static_cast<int>(A.shape[0]);
  int N = static_cast<int>(B.shape[1]);
  int K = static_cast<int>(A.shape[1]);

  int lda = static_cast<int>(A.stride[0]);
  int ldb = static_cast<int>(B.stride[0]);
  int ldc = static_cast<int>(C.stride[0]);

  // CUTLASS bfloat16_t is layout-compatible with __nv_bfloat16.
  auto* a_ptr = reinterpret_cast<cutlass::bfloat16_t*>(A.data);
  auto* b_ptr = reinterpret_cast<cutlass::bfloat16_t*>(B.data);
  auto* c_ptr = reinterpret_cast<cutlass::bfloat16_t*>(C.data);

  GemmBf16 gemm_op;
  GemmBf16::Arguments args(
      {M, N, K},
      {a_ptr, lda},
      {b_ptr, ldb},
      {c_ptr, ldc},
      {c_ptr, ldc},
      {alpha, beta}
  );

  cutlass::Status status = gemm_op(args, nullptr, stream.get());
  return cutlass_to_gpt_status(status);
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

  int batch = static_cast<int>(A.shape[0]);
  int M = static_cast<int>(A.shape[1]);
  int N = static_cast<int>(B.shape[2]);
  int K = static_cast<int>(A.shape[2]);

  int lda = static_cast<int>(A.stride[1]);  // stride of row within a batch
  int ldb = static_cast<int>(B.stride[1]);
  int ldc = static_cast<int>(C.stride[1]);

  int64_t stride_a = A.stride[0];  // stride between batches
  int64_t stride_b = B.stride[0];
  int64_t stride_c = C.stride[0];

  BatchedGemmF32 gemm_op;
  BatchedGemmF32::Arguments args(
      {M, N, K},
      {A.data, lda}, stride_a,
      {B.data, ldb}, stride_b,
      {C.data, ldc}, stride_c,
      {C.data, ldc}, stride_c,
      {alpha, beta},
      batch
  );

  cutlass::Status status = gemm_op(args, nullptr, stream.get());
  return cutlass_to_gpt_status(status);
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

  int batch = static_cast<int>(A.shape[0]);
  int M = static_cast<int>(A.shape[1]);
  int N = static_cast<int>(B.shape[2]);
  int K = static_cast<int>(A.shape[2]);

  int lda = static_cast<int>(A.stride[1]);
  int ldb = static_cast<int>(B.stride[1]);
  int ldc = static_cast<int>(C.stride[1]);

  int64_t stride_a = A.stride[0];
  int64_t stride_b = B.stride[0];
  int64_t stride_c = C.stride[0];

  auto* a_ptr = reinterpret_cast<cutlass::bfloat16_t*>(A.data);
  auto* b_ptr = reinterpret_cast<cutlass::bfloat16_t*>(B.data);
  auto* c_ptr = reinterpret_cast<cutlass::bfloat16_t*>(C.data);

  BatchedGemmBf16 gemm_op;
  BatchedGemmBf16::Arguments args(
      {M, N, K},
      {a_ptr, lda}, stride_a,
      {b_ptr, ldb}, stride_b,
      {c_ptr, ldc}, stride_c,
      {c_ptr, ldc}, stride_c,
      {alpha, beta},
      batch
  );

  cutlass::Status status = gemm_op(args, nullptr, stream.get());
  return cutlass_to_gpt_status(status);
}

}  // namespace kernels
}  // namespace gpt
