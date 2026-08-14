// Layer 4 — Attention subsystem: host-side orchestration.
//
// Wires together linear projection, layout transforms, batched GEMM,
// masking, softmax, and merge operations into a complete attention
// forward/backward pass.

#include "src/attention/attention.h"

#include "src/attention/attention_forward_internal.h"
#include "src/kernels/gemm.h"
#include "src/kernels/layout_kernels.h"
#include "src/ops/linear.h"
#include "src/ops/softmax.h"

namespace gpt {
namespace attention {

Status attention_forward(Tensor3D<const float> X,
                         const AttentionConfig& config,
                         const AttentionParams& params,
                         Tensor3D<float> output,
                         AttentionForwardState& state,
                         const CudaStream& stream) {
  int64_t B = X.shape[0];
  int64_t T = X.shape[1];
  int64_t D = X.shape[2];
  int64_t H = config.n_heads;
  int64_t Dh = config.head_dim;

  if (D != config.d_model) {
    return Status(StatusCode::kInvalidArgument,
                  "attention_forward: D != config.d_model");
  }

  // Step 1: QKV projection — X:[B*T, D] @ W_qkv^T:[D, 3D] → qkv:[B*T, 3D]
  {
    Tensor2D<const float> X_2d;
    X_2d.data = X.data;
    X_2d.shape = {B * T, D};
    X_2d.stride = {D, 1};

    Tensor2D<float> qkv_2d;
    qkv_2d.data = state.qkv.data;
    qkv_2d.shape = {B * T, 3 * D};
    qkv_2d.stride = {3 * D, 1};

    ops::LinearParams lp;
    lp.weight = params.W_qkv;
    lp.bias = params.b_qkv;
    lp.use_bias = config.use_bias;

    GPT_RETURN_IF_ERROR(ops::linear_forward(X_2d, lp, qkv_2d, stream));
  }

  // Step 2: Split Q, K, V and reshape to head layout [B, H, T, Dh].
  // state.qkv is [B, T, 3*D].  Split into three [B, T, D] and repack.
  {
    // QKV is laid out as [B, T, 3, H, Dh] = [B, T, 3*D], with Q/K/V chunks
    // at offsets 0, D, and 2D inside each row. split_heads_f32 reads those
    // strided views and writes contiguous [B, H, T, Dh] buffers.

    // Q
    Tensor3D<const float> Q_flat;
    Q_flat.data = state.qkv.data;  // offset 0
    Q_flat.shape = {B, T, D};
    Q_flat.stride = {T * 3 * D, 3 * D, 1};

    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(Q_flat, state.Q, H, stream));

    Tensor3D<const float> K_flat;
    K_flat.data = state.qkv.data + D;
    K_flat.shape = {B, T, D};
    K_flat.stride = {T * 3 * D, 3 * D, 1};
    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(K_flat, state.K, H, stream));

    Tensor3D<const float> V_flat;
    V_flat.data = state.qkv.data + 2 * D;
    V_flat.shape = {B, T, D};
    V_flat.stride = {T * 3 * D, 3 * D, 1};
    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(V_flat, state.V, H, stream));
  }

  // Step 3-4: Scores = Q @ K^T, optionally fused scale + causal mask.
  // Q: [B*H, T, Dh], K^T: [B*H, Dh, T] → scores: [B*H, T, T]
  {
    Tensor3D<float> Q_3d;
    Q_3d.data = state.Q.data;
    Q_3d.shape = {B * H, T, Dh};
    Q_3d.stride = {T * Dh, Dh, 1};

    Tensor3D<float> K_3d;
    K_3d.data = state.K.data;
    K_3d.shape = {B * H, Dh, T};
    K_3d.stride = {T * Dh, 1, Dh};  // transposed last two dims

    Tensor3D<float> S_3d;
    S_3d.data = state.scores.data;
    S_3d.shape = {B * H, T, T};
    S_3d.stride = {T * T, T, 1};

    float alpha = config.causal ? 1.0f : config.scale();
    GPT_RETURN_IF_ERROR(
        kernels::batched_gemm_f32(Q_3d, K_3d, S_3d, alpha, 0.0f, stream));
    if (config.causal) {
      GPT_RETURN_IF_ERROR(
          scale_and_causal_mask_scores_f32(state.scores, config.scale(), stream));
    }
  }

  // Step 5-6: Causal mask + softmax.
  // Flatten scores to [B*H*T, T] for row-wise softmax.
  {
    Tensor2D<const float> S_2d;
    S_2d.data = state.scores.data;
    S_2d.shape = {B * H * T, T};
    S_2d.stride = {T, 1};

    Tensor2D<const int8_t> null_mask;
    null_mask.data = nullptr;
    null_mask.shape = {B * H * T, T};
    null_mask.stride = {T, 1};

    Tensor2D<float> P_2d;
    P_2d.data = state.probs.data;
    P_2d.shape = {B * H * T, T};
    P_2d.stride = {T, 1};

    GPT_RETURN_IF_ERROR(
        ops::masked_softmax_forward(S_2d, null_mask, P_2d, stream));
  }

  // Step 7: Context = P @ V.
  // P: [B*H, T, T], V: [B*H, T, Dh] → context: [B*H, T, Dh]
  {
    Tensor3D<float> P_3d;
    P_3d.data = state.probs.data;
    P_3d.shape = {B * H, T, T};
    P_3d.stride = {T * T, T, 1};

    Tensor3D<float> V_3d;
    V_3d.data = state.V.data;
    V_3d.shape = {B * H, T, Dh};
    V_3d.stride = {T * Dh, Dh, 1};

    Tensor3D<float> C_3d;
    C_3d.data = state.context.data;
    C_3d.shape = {B * H, T, Dh};
    C_3d.stride = {T * Dh, Dh, 1};

    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        P_3d, V_3d, C_3d, 1.0f, 0.0f, stream));
  }

  // Step 8: Merge heads [B, H, T, Dh] → [B, T, D].
  GPT_RETURN_IF_ERROR(
      kernels::merge_heads_f32(state.context, state.context_merged, stream));

  // Step 9: Output projection.
  {
    Tensor2D<const float> C_2d;
    C_2d.data = state.context_merged.data;
    C_2d.shape = {B * T, D};
    C_2d.stride = {D, 1};

    Tensor2D<float> O_2d;
    O_2d.data = output.data;
    O_2d.shape = {B * T, D};
    O_2d.stride = {D, 1};

    ops::LinearParams lp;
    lp.weight = params.W_o;
    lp.bias = params.b_o;
    lp.use_bias = config.use_bias;

    GPT_RETURN_IF_ERROR(ops::linear_forward(C_2d, lp, O_2d, stream));
  }

  return Status::Ok();
}

Status attention_backward(Tensor3D<const float> dO,
                          Tensor3D<const float> X,
                          const AttentionConfig& config,
                          const AttentionParams& params,
                          const AttentionForwardState& state,
                          Tensor3D<float> dX,
                          AttentionGrads& grads,
                          const CudaStream& stream) {
  // TODO(m5): Implement attention backward.
  //
  // Stages (reverse order of forward):
  //   9. Output projection backward → dC_merged, dW_o, db_o
  //   8. Merge heads backward (= split heads of gradient)
  //   7. Context matmul backward → dP, dV
  //   6. Softmax backward → dS
  //   5. Causal mask backward (masked positions get zero grad)
  //   3-4. Score matmul backward → dQ, dK
  //   2. Merge heads backward for Q,K,V → dQKV_flat
  //   1. QKV projection backward → dX, dW_qkv, db_qkv

  return Status(StatusCode::kNotImplemented,
                "attention_backward not yet implemented");
}

}  // namespace attention
}  // namespace gpt
