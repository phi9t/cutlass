// Layer 4 — Attention subsystem: host-side orchestration.
//
// Wires together linear projection, layout transforms, batched GEMM,
// masking, softmax, and merge operations into a complete attention
// forward/backward pass.

#include "src/attention/attention.h"

#include <cuda_runtime.h>

#include "src/attention/attention_backward_internal.h"
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
                          DeviceScratchArena& scratch,
                          const CudaStream& stream) {
  int64_t B = X.shape[0];
  int64_t T = X.shape[1];
  int64_t D = X.shape[2];
  int64_t H = config.n_heads;
  int64_t Dh = config.head_dim;

  if (D != config.d_model || D != H * Dh) {
    return Status(StatusCode::kInvalidArgument,
                  "attention_backward: invalid model/head dimensions");
  }

  auto d_context_merged_ptr = scratch.alloc_f32(B * T * D);
  if (!d_context_merged_ptr.ok()) return d_context_merged_ptr.status();
  auto d_context_ptr = scratch.alloc_f32(B * H * T * Dh);
  if (!d_context_ptr.ok()) return d_context_ptr.status();
  auto dP_ptr = scratch.alloc_f32(B * H * T * T);
  if (!dP_ptr.ok()) return dP_ptr.status();
  auto dV_ptr = scratch.alloc_f32(B * H * T * Dh);
  if (!dV_ptr.ok()) return dV_ptr.status();
  auto dS_ptr = scratch.alloc_f32(B * H * T * T);
  if (!dS_ptr.ok()) return dS_ptr.status();
  auto dQ_ptr = scratch.alloc_f32(B * H * T * Dh);
  if (!dQ_ptr.ok()) return dQ_ptr.status();
  auto dK_ptr = scratch.alloc_f32(B * H * T * Dh);
  if (!dK_ptr.ok()) return dK_ptr.status();
  auto dQKV_ptr = scratch.alloc_f32(B * T * 3 * D);
  if (!dQKV_ptr.ok()) return dQKV_ptr.status();

  Tensor3D<float> d_context_merged{
      d_context_merged_ptr.value(), {B, T, D}, {T * D, D, 1}};
  Tensor4D<float> d_context{
      d_context_ptr.value(), {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  Tensor4D<float> dP{
      dP_ptr.value(), {B, H, T, T}, {H * T * T, T * T, T, 1}};
  Tensor4D<float> dV{
      dV_ptr.value(), {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  Tensor4D<float> dS{
      dS_ptr.value(), {B, H, T, T}, {H * T * T, T * T, T, 1}};
  Tensor4D<float> dQ{
      dQ_ptr.value(), {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  Tensor4D<float> dK{
      dK_ptr.value(), {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  Tensor3D<float> dQKV{
      dQKV_ptr.value(), {B, T, 3 * D}, {T * 3 * D, 3 * D, 1}};

  // Step 9: output projection backward.
  {
    Tensor2D<const float> C_2d{
        state.context_merged.data, {B * T, D}, {D, 1}};
    Tensor2D<const float> dO_2d{
        dO.data, {B * T, D}, {D, 1}};
    Tensor2D<float> dC_2d{
        d_context_merged.data, {B * T, D}, {D, 1}};

    ops::LinearParams lp;
    lp.weight = params.W_o;
    lp.bias = params.b_o;
    lp.use_bias = config.use_bias;

    ops::LinearGrads linear_grads;
    linear_grads.d_weight = grads.dW_o;
    linear_grads.d_bias = grads.db_o;
    GPT_RETURN_IF_ERROR(
        ops::linear_backward(C_2d, dO_2d, lp, dC_2d, linear_grads, stream));
  }

  // Step 8: merge-heads backward.
  GPT_RETURN_IF_ERROR(
      split_merged_heads_grad_f32(d_context_merged, d_context, H, stream));

  // Step 7: Context = P @ V.
  {
    Tensor3D<float> dC_3d{
        d_context.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
    Tensor3D<float> V_T_3d{
        state.V.data, {B * H, Dh, T}, {T * Dh, 1, Dh}};
    Tensor3D<float> dP_3d{
        dP.data, {B * H, T, T}, {T * T, T, 1}};
    GPT_RETURN_IF_ERROR(
        kernels::batched_gemm_f32(dC_3d, V_T_3d, dP_3d, 1.0f, 0.0f, stream));

    Tensor3D<float> P_T_3d{
        state.probs.data, {B * H, T, T}, {T * T, 1, T}};
    Tensor3D<float> dV_3d{
        dV.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
    GPT_RETURN_IF_ERROR(
        kernels::batched_gemm_f32(P_T_3d, dC_3d, dV_3d, 1.0f, 0.0f, stream));
  }

  // Step 6: softmax backward, then score-scale backward.
  {
    Tensor2D<const float> dP_2d{dP.data, {B * H * T, T}, {T, 1}};
    Tensor2D<const float> P_2d{state.probs.data, {B * H * T, T}, {T, 1}};
    Tensor2D<float> dS_2d{dS.data, {B * H * T, T}, {T, 1}};
    GPT_RETURN_IF_ERROR(
        ops::masked_softmax_backward(dP_2d, P_2d, dS_2d, stream));
    GPT_RETURN_IF_ERROR(scale_scores_grad_f32(dS, config.scale(), stream));
  }

  // Step 3-4: Scores = Q @ K^T.
  {
    Tensor3D<float> dS_3d{
        dS.data, {B * H, T, T}, {T * T, T, 1}};
    Tensor3D<float> K_3d{
        state.K.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
    Tensor3D<float> dQ_3d{
        dQ.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
    GPT_RETURN_IF_ERROR(
        kernels::batched_gemm_f32(dS_3d, K_3d, dQ_3d, 1.0f, 0.0f, stream));

    Tensor3D<float> dS_T_3d{
        dS.data, {B * H, T, T}, {T * T, 1, T}};
    Tensor3D<float> Q_3d{
        state.Q.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
    Tensor3D<float> dK_3d{
        dK.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
    GPT_RETURN_IF_ERROR(
        kernels::batched_gemm_f32(dS_T_3d, Q_3d, dK_3d, 1.0f, 0.0f, stream));
  }

  // Step 2: pack dQ/dK/dV back into flat QKV gradient layout.
  GPT_RETURN_IF_ERROR(merge_qkv_grads_f32(dQ, dK, dV, dQKV, stream));

  // Step 1: QKV projection backward.
  {
    Tensor2D<const float> X_2d{X.data, {B * T, D}, {D, 1}};
    Tensor2D<const float> dQKV_2d{
        dQKV.data, {B * T, 3 * D}, {3 * D, 1}};
    Tensor2D<float> dX_2d{dX.data, {B * T, D}, {D, 1}};

    ops::LinearParams lp;
    lp.weight = params.W_qkv;
    lp.bias = params.b_qkv;
    lp.use_bias = config.use_bias;

    ops::LinearGrads linear_grads;
    linear_grads.d_weight = grads.dW_qkv;
    linear_grads.d_bias = grads.db_qkv;
    GPT_RETURN_IF_ERROR(
        ops::linear_backward(X_2d, dQKV_2d, lp, dX_2d, linear_grads, stream));
  }

  return Status::Ok();
}

}  // namespace attention
}  // namespace gpt
