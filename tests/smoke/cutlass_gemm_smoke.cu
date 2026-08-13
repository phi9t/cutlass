/***************************************************************************************************
 * Infra smoke test — single-GPU CUTLASS GEMM.
 *
 * This is NOT feature work. It is the toolchain proof for the fork's hermetic
 * CUDA build (ADR 0002/0003): it depends ONLY on the real upstream CUTLASS
 * headers (//include:cutlass) and the CUDA runtime, so a passing run proves the
 * rootfs nvcc compiled a CUTLASS device kernel for the local GPU's arch, linked
 * cudart, and produced numerically correct output on one GPU. It deliberately
 * does not touch the //src fork trainer scaffold, whose kernels are out of scope
 * until infra lands.
 *
 * Structure is adapted from examples/00_basic_gemm: an SGEMM (column-major,
 * NN) launched via cutlass::gemm::device::Gemm, verified bit-exact against a
 * naive reference kernel. Distinct exit codes let the verifier ladder tell a
 * CUDA-runtime failure apart from a correctness mismatch.
 **************************************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/gemm/device/gemm.h"

namespace {

// Exit codes distinct from the shell ladder's build/preflight codes so a
// non-zero here is unambiguously "GEMM smoke ran but failed".
enum SmokeExit {
  kOk = 0,
  kCudaError = 3,       // a CUDA runtime call or the CUTLASS launch failed
  kResultMismatch = 4,  // CUTLASS output disagreed with the reference kernel
};

#define CHECK_CUDA(expr)                                                    \
  do {                                                                      \
    cudaError_t err_ = (expr);                                              \
    if (err_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                              \
      return kCudaError;                                                    \
    }                                                                       \
  } while (0)

// Column-major single-precision NN GEMM via the CUTLASS device API. Uses the
// default kernel configuration, which the library specializes for the compiled
// architecture (sm_100 inside the rootfs).
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const* A,
                           int lda, float const* B, int ldb, float beta,
                           float* C, int ldc) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float,
                                           ColumnMajor, float, ColumnMajor>;
  Gemm op;
  Gemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc},
                       {alpha, beta});
  if (op(args) != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

// Fill a column-major matrix with small deterministic integers.
__global__ void InitMatrix(float* m, int rows, int cols, int seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < rows && j < cols) {
    int offset = i + j * rows;
    int const k = 16807;
    int const mod = 16;
    m[offset] = float(((offset + seed) * k % mod) - mod / 2);
  }
}

// Naive column-major reference GEMM used as the correctness oracle.
__global__ void ReferenceGemm(int M, int N, int K, float alpha, float const* A,
                              int lda, float const* B, int ldb, float beta,
                              float* C, int ldc) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < M && j < N) {
    float acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += A[i + k * lda] * B[k + j * ldb];
    }
    C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
  }
}

cudaError_t Allocate(float** m, int rows, int cols, int seed) {
  size_t bytes = sizeof(float) * rows * cols;
  cudaError_t r = cudaMalloc(reinterpret_cast<void**>(m), bytes);
  if (r != cudaSuccess) return r;
  r = cudaMemset(*m, 0, bytes);
  if (r != cudaSuccess) return r;
  dim3 block(16, 16);
  dim3 grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y);
  InitMatrix<<<grid, block>>>(*m, rows, cols, seed);
  return cudaGetLastError();
}

int RunSmoke(int M, int N, int K) {
  float const alpha = 1.0f;
  float const beta = 0.0f;
  int const lda = M, ldb = K, ldc = M;
  size_t const sizeof_C = sizeof(float) * ldc * N;

  float *A = nullptr, *B = nullptr, *C_cutlass = nullptr, *C_ref = nullptr;
  CHECK_CUDA(Allocate(&A, M, K, 0));
  CHECK_CUDA(Allocate(&B, K, N, 17));
  CHECK_CUDA(Allocate(&C_cutlass, M, N, 101));
  CHECK_CUDA(Allocate(&C_ref, M, N, 101));
  CHECK_CUDA(cudaMemcpy(C_ref, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice));

  cudaError_t launch = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta,
                                      C_cutlass, ldc);
  if (launch != cudaSuccess) {
    std::fprintf(stderr, "CUTLASS GEMM launch failed: %s\n",
                 cudaGetErrorString(launch));
    return kCudaError;
  }

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  ReferenceGemm<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C_ref,
                                 ldc);
  CHECK_CUDA(cudaGetLastError());

  std::vector<float> host_cutlass(ldc * N, 0);
  std::vector<float> host_ref(ldc * N, 0);
  CHECK_CUDA(cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(host_ref.data(), C_ref, sizeof_C,
                        cudaMemcpyDeviceToHost));

  cudaFree(C_ref);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  if (host_cutlass != host_ref) {
    std::fprintf(stderr, "CUTLASS result mismatch vs reference GEMM\n");
    return kResultMismatch;
  }
  return kOk;
}

}  // namespace

// usage: cutlass_gemm_smoke [M N K]  (defaults 256x256x256)
int main(int argc, char const* argv[]) {
  int dims[3] = {256, 256, 256};
  for (int i = 1; i < argc && i < 4; ++i) {
    dims[i - 1] = std::atoi(argv[i]);
  }

  cudaDeviceProp prop;
  cudaError_t r = cudaGetDeviceProperties(&prop, 0);
  if (r != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(r));
    return kCudaError;
  }
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  int rc = RunSmoke(dims[0], dims[1], dims[2]);
  if (rc == kOk) {
    std::printf("cutlass_gemm_smoke: PASS (%dx%dx%d)\n", dims[0], dims[1],
                dims[2]);
  }
  return rc;
}
