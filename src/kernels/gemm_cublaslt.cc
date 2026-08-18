// Layer 2 — Primitive kernels: cuBLASLt GEMM fallback backend.
//
// Used for correctness isolation and backend comparison against the CUTLASS
// path. The first slice keeps descriptors per-call and supports fp32 GEMM.

#include "src/kernels/gemm_cublaslt_internal.h"

#include <string>

#include <cublasLt.h>

#include "src/kernels/gemm_problem.h"

namespace gpt {
namespace kernels {

namespace {

Status cublas_status(cublasStatus_t status, const char* context) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Status::Ok();
  }
  return Status(StatusCode::kCudaError,
                std::string(context) + " failed with cuBLAS status " +
                    std::to_string(static_cast<int>(status)));
}

Status create_layout(cublasLtMatrixLayout_t* layout, cudaDataType type,
                     int64_t rows, int64_t cols, int64_t leading_dim,
                     MatrixOrientation orientation) {
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutCreate(layout, type, static_cast<uint64_t>(rows),
                                 static_cast<uint64_t>(cols), leading_dim),
      "cublasLtMatrixLayoutCreate"));

  cublasLtOrder_t cublas_order =
      (orientation == MatrixOrientation::kRowMajor) ? CUBLASLT_ORDER_ROW
                                                    : CUBLASLT_ORDER_COL;
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &cublas_order, sizeof(cublas_order)),
      "cublasLtMatrixLayoutSetAttribute(order)"));
  return Status::Ok();
}

Status set_batch_layout(cublasLtMatrixLayout_t layout, int32_t batch_count,
                        int64_t batch_stride) {
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                       &batch_count, sizeof(batch_count)),
      "cublasLtMatrixLayoutSetAttribute(batch_count)"));
  GPT_RETURN_IF_ERROR(cublas_status(
      cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
          sizeof(batch_stride)),
      "cublasLtMatrixLayoutSetAttribute(strided_batch_offset)"));
  return Status::Ok();
}

void destroy_layout(cublasLtMatrixLayout_t layout) {
  if (layout != nullptr) {
    cublasLtMatrixLayoutDestroy(layout);
  }
}

template <typename T>
Status gemm_cublaslt_impl(Tensor2D<T> A, Tensor2D<T> B, Tensor2D<T> C,
                          const GemmProblem& problem,
                          cudaDataType type, const char* context, float alpha,
                          float beta, const CudaStream& stream) {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t matmul = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasStatus_t status = cublasLtCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return cublas_status(status, "cublasLtCreate");
  }

  auto cleanup = [&]() {
    destroy_layout(c_layout);
    destroy_layout(b_layout);
    destroy_layout(a_layout);
    if (matmul != nullptr) {
      cublasLtMatmulDescDestroy(matmul);
    }
    if (handle != nullptr) {
      cublasLtDestroy(handle);
    }
  };

  Status result = cublas_status(
      cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  if (!result.ok()) {
    cleanup();
    return result;
  }

  result = create_layout(&a_layout, type, A.shape[0], A.shape[1],
                         problem.a.leading_dim, problem.a.orientation);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  result = create_layout(&b_layout, type, B.shape[0], B.shape[1],
                         problem.b.leading_dim, problem.b.orientation);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  result = create_layout(&c_layout, type, C.shape[0], C.shape[1],
                         problem.c.leading_dim, problem.c.orientation);
  if (!result.ok()) {
    cleanup();
    return result;
  }
  if (problem.batch_count > 1) {
    int32_t batch_count = static_cast<int32_t>(problem.batch_count);
    result = set_batch_layout(a_layout, batch_count, problem.batch_stride_a);
    if (!result.ok()) {
      cleanup();
      return result;
    }
    result = set_batch_layout(b_layout, batch_count, problem.batch_stride_b);
    if (!result.ok()) {
      cleanup();
      return result;
    }
    result = set_batch_layout(c_layout, batch_count, problem.batch_stride_c);
    if (!result.ok()) {
      cleanup();
      return result;
    }
  }

  status = cublasLtMatmul(handle, matmul, &alpha, A.data, a_layout, B.data,
                          b_layout, &beta, C.data, c_layout, C.data, c_layout,
                          nullptr, nullptr, 0, stream.get());
  result = cublas_status(status, "cublasLtMatmul");
  cleanup();
  if (!result.ok()) {
    return result;
  }
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

template <typename T>
Tensor2D<T> batch_slice(Tensor3D<T> tensor) {
  return Tensor2D<T>{tensor.data, {tensor.shape[1], tensor.shape[2]},
                     {tensor.stride[1], tensor.stride[2]}};
}

}  // namespace

Status gemm_f32_cublaslt(Tensor2D<float> A, Tensor2D<float> B,
                         Tensor2D<float> C, float alpha, float beta,
                         const CudaStream& stream) {
  Result<GemmProblem> problem = make_gemm_problem(A, B, C);
  if (!problem.ok()) {
    return problem.status();
  }
  return gemm_cublaslt_impl(A, B, C, problem.value(), CUDA_R_32F,
                            "gemm_f32_cublaslt", alpha, beta, stream);
}

Status gemm_bf16_cublaslt(Tensor2D<__nv_bfloat16> A,
                          Tensor2D<__nv_bfloat16> B,
                          Tensor2D<__nv_bfloat16> C, float alpha, float beta,
                          const CudaStream& stream) {
  Result<GemmProblem> problem = make_gemm_problem(A, B, C);
  if (!problem.ok()) {
    return problem.status();
  }
  return gemm_cublaslt_impl(A, B, C, problem.value(), CUDA_R_16BF,
                            "gemm_bf16_cublaslt", alpha, beta, stream);
}

Status batched_gemm_f32_cublaslt(Tensor3D<float> A, Tensor3D<float> B,
                                 Tensor3D<float> C, float alpha, float beta,
                                 const CudaStream& stream) {
  Result<GemmProblem> problem = make_batched_gemm_problem(A, B, C);
  if (!problem.ok()) {
    return problem.status();
  }
  return gemm_cublaslt_impl(batch_slice(A), batch_slice(B), batch_slice(C),
                            problem.value(), CUDA_R_32F,
                            "batched_gemm_f32_cublaslt", alpha, beta, stream);
}

Status batched_gemm_bf16_cublaslt(Tensor3D<__nv_bfloat16> A,
                                  Tensor3D<__nv_bfloat16> B,
                                  Tensor3D<__nv_bfloat16> C, float alpha,
                                  float beta, const CudaStream& stream) {
  Result<GemmProblem> problem = make_batched_gemm_problem(A, B, C);
  if (!problem.ok()) {
    return problem.status();
  }
  return gemm_cublaslt_impl(batch_slice(A), batch_slice(B), batch_slice(C),
                            problem.value(), CUDA_R_16BF,
                            "batched_gemm_bf16_cublaslt", alpha, beta, stream);
}

}  // namespace kernels
}  // namespace gpt
