// Layer 4 — Attention subsystem: host-side orchestration.
//
// Wires together linear projection, layout transforms, batched GEMM,
// masking, softmax, and merge operations into a complete attention
// forward/backward pass.

#include "src/attention/attention.h"

#include <cuda_runtime.h>

#include <utility>

#include "src/attention/attention_backward_internal.h"
#include "src/attention/attention_forward_internal.h"
#include "src/kernels/gemm.h"
#include "src/kernels/layout_kernels.h"
#include "src/ops/linear.h"
#include "src/ops/softmax.h"

namespace gpt {
namespace attention {

namespace {

Status AllocF32(int64_t count, float** ptr) {
  cudaError_t err =
      cudaMalloc(ptr, static_cast<size_t>(count) * sizeof(float));
  if (err != cudaSuccess) {
    *ptr = nullptr;
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  return Status::Ok();
}

Tensor3D<float> Make3D(float* ptr, int64_t B, int64_t T, int64_t D) {
  return Tensor3D<float>{ptr, {B, T, D}, {T * D, D, 1}};
}

Tensor4D<float> Make4D(float* ptr, int64_t B, int64_t H, int64_t T,
                       int64_t Dh) {
  return Tensor4D<float>{ptr, {B, H, T, Dh},
                         {H * T * Dh, T * Dh, Dh, 1}};
}

struct HeadViews {
  Tensor3D<const float> q_flat;
  Tensor3D<const float> k_flat;
  Tensor3D<const float> v_flat;
  Tensor3D<float> q_batched;
  Tensor3D<float> k_batched;
  Tensor3D<float> k_transposed_batched;
  Tensor3D<float> v_batched;
  Tensor3D<float> scores_batched;
  Tensor3D<float> probs_batched;
  Tensor3D<float> context_batched;
};

HeadViews MakeHeadViews(AttentionForwardState& state, int64_t B, int64_t T,
                        int64_t D, int64_t H, int64_t Dh) {
  HeadViews views;
  views.q_flat = {state.qkv.data, {B, T, D}, {T * 3 * D, 3 * D, 1}};
  views.k_flat = {state.qkv.data + D, {B, T, D}, {T * 3 * D, 3 * D, 1}};
  views.v_flat = {state.qkv.data + 2 * D, {B, T, D},
                  {T * 3 * D, 3 * D, 1}};
  views.q_batched = {state.Q.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
  views.k_batched = {state.K.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
  views.k_transposed_batched =
      Tensor3D<float>{state.K.data, {B * H, Dh, T}, {T * Dh, 1, Dh}};
  views.v_batched = {state.V.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
  views.scores_batched =
      {state.scores.data, {B * H, T, T}, {T * T, T, 1}};
  views.probs_batched = {state.probs.data, {B * H, T, T}, {T * T, T, 1}};
  views.context_batched =
      {state.context.data, {B * H, T, Dh}, {T * Dh, Dh, 1}};
  return views;
}

}  // namespace

Status AttentionWorkspace::ensure(const AttentionConfig& config, int64_t B,
                                  int64_t T) {
  if (B <= 0 || T <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "AttentionWorkspace::ensure: empty batch");
  }
  if (config.d_model <= 0 || config.n_heads <= 0 || config.head_dim <= 0 ||
      config.n_heads * config.head_dim != config.d_model) {
    return Status(StatusCode::kInvalidArgument,
                  "AttentionWorkspace::ensure: invalid attention shape");
  }
  if (capacity_B_ == B && capacity_T_ == T &&
      capacity_d_model_ == config.d_model &&
      capacity_n_heads_ == config.n_heads &&
      capacity_head_dim_ == config.head_dim) {
    return Status::Ok();
  }

  release();

  const int64_t D = config.d_model;
  const int64_t H = config.n_heads;
  const int64_t Dh = config.head_dim;

  auto add = [this](int64_t count) -> Result<float*> {
    float* ptr = nullptr;
    Status status = AllocF32(count, &ptr);
    if (!status.ok()) return status;
    buffers_.push_back(ptr);
    return ptr;
  };

  auto qkv = add(B * T * 3 * D);
  if (!qkv.ok()) return qkv.status();
  auto Q = add(B * H * T * Dh);
  if (!Q.ok()) return Q.status();
  auto K = add(B * H * T * Dh);
  if (!K.ok()) return K.status();
  auto V = add(B * H * T * Dh);
  if (!V.ok()) return V.status();
  auto scores = add(B * H * T * T);
  if (!scores.ok()) return scores.status();
  auto probs = add(B * H * T * T);
  if (!probs.ok()) return probs.status();
  auto ctx = add(B * H * T * Dh);
  if (!ctx.ok()) return ctx.status();
  auto merged = add(B * T * D);
  if (!merged.ok()) return merged.status();

  state_.qkv = Make3D(qkv.value(), B, T, 3 * D);
  state_.Q = Make4D(Q.value(), B, H, T, Dh);
  state_.K = Make4D(K.value(), B, H, T, Dh);
  state_.V = Make4D(V.value(), B, H, T, Dh);
  state_.scores = Tensor4D<float>{scores.value(), {B, H, T, T},
                                  {H * T * T, T * T, T, 1}};
  state_.probs = Tensor4D<float>{probs.value(), {B, H, T, T},
                                 {H * T * T, T * T, T, 1}};
  state_.context = Make4D(ctx.value(), B, H, T, Dh);
  state_.context_merged = Make3D(merged.value(), B, T, D);
  capacity_B_ = B;
  capacity_T_ = T;
  capacity_d_model_ = config.d_model;
  capacity_n_heads_ = config.n_heads;
  capacity_head_dim_ = config.head_dim;
  return Status::Ok();
}

void AttentionWorkspace::release() {
  for (float* ptr : buffers_) {
    cudaFree(ptr);
  }
  buffers_.clear();
  state_ = AttentionForwardState{};
  capacity_B_ = 0;
  capacity_T_ = 0;
  capacity_d_model_ = 0;
  capacity_n_heads_ = 0;
  capacity_head_dim_ = 0;
}

AttentionWorkspace::~AttentionWorkspace() { release(); }

AttentionWorkspace::AttentionWorkspace(AttentionWorkspace&& other) noexcept
    : state_(other.state_),
      buffers_(std::move(other.buffers_)),
      capacity_B_(other.capacity_B_),
      capacity_T_(other.capacity_T_),
      capacity_d_model_(other.capacity_d_model_),
      capacity_n_heads_(other.capacity_n_heads_),
      capacity_head_dim_(other.capacity_head_dim_) {
  other.state_ = AttentionForwardState{};
  other.capacity_B_ = 0;
  other.capacity_T_ = 0;
  other.capacity_d_model_ = 0;
  other.capacity_n_heads_ = 0;
  other.capacity_head_dim_ = 0;
}

AttentionWorkspace& AttentionWorkspace::operator=(
    AttentionWorkspace&& other) noexcept {
  if (this != &other) {
    release();
    state_ = other.state_;
    buffers_ = std::move(other.buffers_);
    capacity_B_ = other.capacity_B_;
    capacity_T_ = other.capacity_T_;
    capacity_d_model_ = other.capacity_d_model_;
    capacity_n_heads_ = other.capacity_n_heads_;
    capacity_head_dim_ = other.capacity_head_dim_;
    other.state_ = AttentionForwardState{};
    other.capacity_B_ = 0;
    other.capacity_T_ = 0;
    other.capacity_d_model_ = 0;
    other.capacity_n_heads_ = 0;
    other.capacity_head_dim_ = 0;
  }
  return *this;
}

Status attention_forward_with_state(Tensor3D<const float> X,
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
    HeadViews heads = MakeHeadViews(state, B, T, D, H, Dh);
    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(heads.q_flat, state.Q, H, stream));
    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(heads.k_flat, state.K, H, stream));
    GPT_RETURN_IF_ERROR(
        kernels::split_heads_f32(heads.v_flat, state.V, H, stream));
  }

  // Step 3-4: Scores = Q @ K^T, optionally fused scale + causal mask.
  // Q: [B*H, T, Dh], K^T: [B*H, Dh, T] → scores: [B*H, T, T]
  {
    HeadViews heads = MakeHeadViews(state, B, T, D, H, Dh);
    float alpha = config.causal ? 1.0f : config.scale();
    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        heads.q_batched, heads.k_transposed_batched, heads.scores_batched,
        alpha, 0.0f, stream));
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
    HeadViews heads = MakeHeadViews(state, B, T, D, H, Dh);
    GPT_RETURN_IF_ERROR(kernels::batched_gemm_f32(
        heads.probs_batched, heads.v_batched, heads.context_batched, 1.0f,
        0.0f, stream));
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

Status attention_forward(Tensor3D<const float> X,
                         const AttentionConfig& config,
                         const AttentionParams& params,
                         Tensor3D<float> output,
                         AttentionWorkspace& workspace,
                         const CudaStream& stream) {
  GPT_RETURN_IF_ERROR(workspace.ensure(config, X.shape[0], X.shape[1]));
  return attention_forward_with_state(X, config, params, output,
                                      workspace.state(), stream);
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
