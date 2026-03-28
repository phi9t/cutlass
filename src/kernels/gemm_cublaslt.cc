// Layer 2 — Primitive kernels: cuBLASLt GEMM fallback backend.
//
// Used for early bring-up and correctness isolation before the CUTLASS
// path is fully tuned.

#include "src/kernels/gemm.h"

// TODO(m3): Implement cuBLASLt GEMM wrappers.
//   - Create/cache cuBLASLt handle per device.
//   - Map TensorView strides to cuBLASLt matrix layout descriptors.
//   - Support fp32 and bf16 with fp32 accumulation.
//   - Wire into the GemmBackend::kCublasLt path.

namespace gpt {
namespace kernels {

// cuBLASLt implementations will be added here.
// For now, all calls through GemmBackend::kCublasLt return kNotImplemented
// from the CUTLASS .cu file (the backend switch is in the caller).

}  // namespace kernels
}  // namespace gpt
