// Layer 4 — Attention subsystem: host-side orchestration.
//
// Wires together linear projection, layout transforms, batched GEMM,
// masking, softmax, and merge operations into a complete attention
// forward/backward pass.

#include "src/attention/attention.h"
#include "src/attention/attention_forward.h"

#include "src/kernels/elementwise.h"
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
  // Fused kernel reads packed [B, T, 3*D] and writes Q, K, V as [B, H, T, Dh].
  {
    Tensor3D<const float> qkv_view;
    qkv_view.data = state.qkv.data;
    qkv_view.shape = {B, T, 3 * D};
    qkv_view.stride = {T * 3 * D, 3 * D, 1};

    GPT_RETURN_IF_ERROR(
        kernels::split_qkv_heads_f32(qkv_view, state.Q, state.K, state.V,
                                     H, stream));
  }

  // Step 3-4: Scores = Q @ K^T * scale.
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

    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        Q_3d, K_3d, S_3d, config.scale(), 0.0f, stream));
  }

  // Step 5: Apply causal mask (if enabled).
  if (config.causal) {
    GPT_RETURN_IF_ERROR(
        apply_causal_mask_f32(state.scores.data, B, H, T, stream));
  }

  // Step 6: Softmax over scores → probabilities.
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
  int64_t B = X.shape[0];
  int64_t T = X.shape[1];
  int64_t D = X.shape[2];
  int64_t H = config.n_heads;
  int64_t Dh = config.head_dim;

  // -----------------------------------------------------------------------
  // Stage 9: Output projection backward.
  //   Forward was: O = C_merged @ W_o^T + b_o
  //   Backward:    dC_merged = dO @ W_o
  //                dW_o = dO^T @ C_merged
  //                db_o = col_sum(dO)
  // -----------------------------------------------------------------------
  // Reuse state.context_merged buffer for dC_merged (no longer needed for fwd).
  Tensor3D<float> dC_merged;
  dC_merged.data = state.context_merged.data;
  dC_merged.shape = {B, T, D};
  dC_merged.stride = {T * D, D, 1};

  {
    Tensor2D<const float> dO_2d;
    dO_2d.data = dO.data;
    dO_2d.shape = {B * T, D};
    dO_2d.stride = {D, 1};

    Tensor2D<const float> C_2d;
    C_2d.data = state.context_merged.data;
    C_2d.shape = {B * T, D};
    C_2d.stride = {D, 1};

    Tensor2D<float> dC_2d;
    dC_2d.data = dC_merged.data;
    dC_2d.shape = {B * T, D};
    dC_2d.stride = {D, 1};

    ops::LinearParams lp;
    lp.weight = params.W_o;
    lp.bias = params.b_o;
    lp.use_bias = config.use_bias;

    ops::LinearGrads lg;
    lg.d_weight = grads.dW_o;
    lg.d_bias = grads.db_o;

    GPT_RETURN_IF_ERROR(
        ops::linear_backward(C_2d, dO_2d, lp, dC_2d, lg, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 8: Merge heads backward (= split heads of gradient).
  //   Forward was: merge_heads [B, H, T, Dh] → [B, T, D]
  //   Backward:    split_heads [B, T, D] → [B, H, T, Dh]
  // -----------------------------------------------------------------------
  // Reuse state.context buffer for dContext [B, H, T, Dh].
  Tensor4D<float> dContext;
  dContext.data = state.context.data;
  dContext.shape = {B, H, T, Dh};
  dContext.stride = {H * T * Dh, T * Dh, Dh, 1};

  {
    Tensor3D<const float> dC_in;
    dC_in.data = dC_merged.data;
    dC_in.shape = {B, T, D};
    dC_in.stride = {T * D, D, 1};
    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(dC_in, dContext, H, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 7: Context matmul backward.
  //   Forward was: C = P @ V  (batched: [B*H, T, T] @ [B*H, T, Dh])
  //   Backward:    dP = dC @ V^T
  //                dV = P^T @ dC
  // -----------------------------------------------------------------------
  // Reuse state.scores buffer for dP [B, H, T, T].
  Tensor4D<float> dP;
  dP.data = state.scores.data;
  dP.shape = {B, H, T, T};
  dP.stride = {H * T * T, T * T, T, 1};

  {
    // dP = dC @ V^T
    Tensor3D<float> dC_3d;
    dC_3d.data = dContext.data;
    dC_3d.shape = {B * H, T, Dh};
    dC_3d.stride = {T * Dh, Dh, 1};

    Tensor3D<float> V_T;
    V_T.data = state.V.data;
    V_T.shape = {B * H, Dh, T};
    V_T.stride = {T * Dh, 1, Dh};  // transposed last two dims

    Tensor3D<float> dP_3d;
    dP_3d.data = dP.data;
    dP_3d.shape = {B * H, T, T};
    dP_3d.stride = {T * T, T, 1};

    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        dC_3d, V_T, dP_3d, 1.0f, 0.0f, stream));

    // dV = P^T @ dC
    // Reuse state.V buffer for dV.
    Tensor3D<float> P_T;
    P_T.data = state.probs.data;
    P_T.shape = {B * H, T, T};
    P_T.stride = {T * T, 1, T};  // transposed last two dims

    Tensor3D<float> dV_3d;
    dV_3d.data = state.V.data;
    dV_3d.shape = {B * H, T, Dh};
    dV_3d.stride = {T * Dh, Dh, 1};

    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        P_T, dC_3d, dV_3d, 1.0f, 0.0f, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 6: Softmax backward.
  //   dS = P * (dP - sum(P * dP))   (row-wise)
  // -----------------------------------------------------------------------
  // Reuse dP buffer in-place (overwrite with dS).
  {
    Tensor2D<const float> dP_2d;
    dP_2d.data = dP.data;
    dP_2d.shape = {B * H * T, T};
    dP_2d.stride = {T, 1};

    Tensor2D<const float> P_2d;
    P_2d.data = state.probs.data;
    P_2d.shape = {B * H * T, T};
    P_2d.stride = {T, 1};

    Tensor2D<float> dS_2d;
    dS_2d.data = dP.data;  // overwrite dP with dS
    dS_2d.shape = {B * H * T, T};
    dS_2d.stride = {T, 1};

    GPT_RETURN_IF_ERROR(
        ops::masked_softmax_backward(dP_2d, P_2d, dS_2d, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 5: Causal mask backward — masked positions already have zero
  // probability, so softmax backward naturally zeroes those gradients.
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  // Stages 3-4: Score matmul backward.
  //   Forward was: S = Q @ K^T * scale
  //   Backward:    dQ = dS @ K * scale
  //                dK = dS^T @ Q * scale
  // -----------------------------------------------------------------------
  // Both dQ and dK read from the other's buffer (dQ reads K, dK reads Q),
  // and we want to write dQ→state.Q, dK→state.K. Since writing either one
  // destroys the original that the other needs, we write dQ to a temp buffer
  // (state.context), then copy it to state.Q after dK is computed.
  {
    Tensor3D<float> dS_3d;
    dS_3d.data = dP.data;
    dS_3d.shape = {B * H, T, T};
    dS_3d.stride = {T * T, T, 1};

    Tensor3D<float> K_3d;
    K_3d.data = state.K.data;
    K_3d.shape = {B * H, T, Dh};
    K_3d.stride = {T * Dh, Dh, 1};

    Tensor3D<float> Q_3d;
    Q_3d.data = state.Q.data;
    Q_3d.shape = {B * H, T, Dh};
    Q_3d.stride = {T * Dh, Dh, 1};

    Tensor3D<float> dS_T;
    dS_T.data = dP.data;
    dS_T.shape = {B * H, T, T};
    dS_T.stride = {T * T, 1, T};  // transposed

    // dQ = dS @ K * scale → temp buffer (state.context)
    Tensor3D<float> dQ_temp;
    dQ_temp.data = state.context.data;
    dQ_temp.shape = {B * H, T, Dh};
    dQ_temp.stride = {T * Dh, Dh, 1};

    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        dS_3d, K_3d, dQ_temp, config.scale(), 0.0f, stream));

    // dK = dS^T @ Q * scale → state.K (original K destroyed, but no longer needed)
    Tensor3D<float> dK_3d;
    dK_3d.data = state.K.data;
    dK_3d.shape = {B * H, T, Dh};
    dK_3d.stride = {T * Dh, Dh, 1};

    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        dS_T, Q_3d, dK_3d, config.scale(), 0.0f, stream));

    // Copy dQ from temp to state.Q.
    int64_t n = B * H * T * Dh;
    Tensor1D<const float> src{dQ_temp.data, {n}, {1}};
    Tensor1D<float> dst{state.Q.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_scale_f32(src, 1.0f, dst, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 2: Repack dQ, dK, dV from head layout back into dQKV [B, T, 3*D].
  //   Forward did: split_heads on Q, K, V.
  //   Backward: merge_heads on dQ, dK, dV back into their QKV slots.
  // -----------------------------------------------------------------------
  // dQ is in state.Q [B,H,T,Dh], dK in state.K, dV in state.V.
  // state.qkv [B,T,3*D] will hold dQKV.
  {
    // Merge dQ → dQKV[:, :, 0:D]
    Tensor4D<const float> dQ_4d;
    dQ_4d.data = state.Q.data;
    dQ_4d.shape = {B, H, T, Dh};
    dQ_4d.stride = {H * T * Dh, T * Dh, Dh, 1};

    // We need to merge into strided slots of qkv buffer.
    // merge_heads writes contiguous [B, T, D]. We'll merge each into
    // a temp contiguous block, then scatter into qkv.
    // For v1, merge directly — merge_heads outputs [B, T, D] contiguous.
    // Then we need to interleave into [B, T, 3*D].
    // Simpler: merge dQ into qkv at offset 0 with stride 3*D per row element.
    // But merge_heads_f32 outputs contiguous [B, T, D].
    // So merge to a temp then copy strided. We can reuse context_merged.

    // Merge dQ → context_merged (temp)
    Tensor3D<float> dQ_merged;
    dQ_merged.data = state.context_merged.data;
    dQ_merged.shape = {B, T, D};
    dQ_merged.stride = {T * D, D, 1};
    GPT_RETURN_IF_ERROR(
        kernels::merge_heads_f32(dQ_4d, dQ_merged, stream));

    // Copy into qkv[:, :, 0:D] (strided).
    // qkv is [B, T, 3*D] contiguous. Q slot is at offset 0 with inner stride 1
    // but row stride 3*D.
    Tensor2D<const float> src_2d;
    src_2d.data = dQ_merged.data;
    src_2d.shape = {B * T, D};
    src_2d.stride = {D, 1};

    Tensor2D<float> dst_2d;
    dst_2d.data = state.qkv.data;  // offset 0 for Q
    dst_2d.shape = {B * T, D};
    dst_2d.stride = {3 * D, 1};  // strided into QKV

    GPT_RETURN_IF_ERROR(
        kernels::repack_contiguous_f32(src_2d, dst_2d, stream));

    // Merge dK → context_merged (temp), then copy to qkv[:, :, D:2D]
    Tensor4D<const float> dK_4d;
    dK_4d.data = state.K.data;
    dK_4d.shape = {B, H, T, Dh};
    dK_4d.stride = {H * T * Dh, T * Dh, Dh, 1};

    GPT_RETURN_IF_ERROR(
        kernels::merge_heads_f32(dK_4d, dQ_merged, stream));  // reuse temp

    src_2d.data = dQ_merged.data;
    dst_2d.data = state.qkv.data + D;  // offset D for K
    dst_2d.stride = {3 * D, 1};
    GPT_RETURN_IF_ERROR(
        kernels::repack_contiguous_f32(src_2d, dst_2d, stream));

    // Merge dV → context_merged (temp), then copy to qkv[:, :, 2D:3D]
    Tensor4D<const float> dV_4d;
    dV_4d.data = state.V.data;
    dV_4d.shape = {B, H, T, Dh};
    dV_4d.stride = {H * T * Dh, T * Dh, Dh, 1};

    GPT_RETURN_IF_ERROR(
        kernels::merge_heads_f32(dV_4d, dQ_merged, stream));  // reuse temp

    src_2d.data = dQ_merged.data;
    dst_2d.data = state.qkv.data + 2 * D;  // offset 2D for V
    dst_2d.stride = {3 * D, 1};
    GPT_RETURN_IF_ERROR(
        kernels::repack_contiguous_f32(src_2d, dst_2d, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 1: QKV projection backward.
  //   Forward was: QKV = X @ W_qkv^T + b_qkv
  //   Backward:    dX     = dQKV @ W_qkv
  //                dW_qkv = dQKV^T @ X
  //                db_qkv = col_sum(dQKV)
  // -----------------------------------------------------------------------
  {
    Tensor2D<const float> X_2d;
    X_2d.data = X.data;
    X_2d.shape = {B * T, D};
    X_2d.stride = {D, 1};

    Tensor2D<const float> dQKV_2d;
    dQKV_2d.data = state.qkv.data;
    dQKV_2d.shape = {B * T, 3 * D};
    dQKV_2d.stride = {3 * D, 1};

    Tensor2D<float> dX_2d;
    dX_2d.data = dX.data;
    dX_2d.shape = {B * T, D};
    dX_2d.stride = {D, 1};

    ops::LinearParams lp;
    lp.weight = params.W_qkv;
    lp.bias = params.b_qkv;
    lp.use_bias = config.use_bias;

    ops::LinearGrads lg;
    lg.d_weight = grads.dW_qkv;
    lg.d_bias = grads.db_qkv;

    GPT_RETURN_IF_ERROR(
        ops::linear_backward(X_2d, dQKV_2d, lp, dX_2d, lg, stream));
  }

  return Status::Ok();
}

}  // namespace attention
}  // namespace gpt
