// Tier C — Integration: checkpoint save/load produces identical model.
//
// Initialize a tiny GPT model, save checkpoint, load into new buffers,
// verify identical forward output on the same input.

#include "gtest/gtest.h"

#include "src/checkpoint/checkpoint.h"
#include "src/core/stream.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_model.h"
#include "src/model/gpt_params.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace gpt;
using namespace gpt::model;

namespace {

// Helper to allocate forward state buffers on GPU and wire into fwd_state.
float* allocate_forward_state(const GPTConfig& cfg, int64_t B, int64_t T,
                              GPTForwardState& fwd) {
  int64_t D = cfg.d_model;
  int64_t H = cfg.n_heads;
  int64_t Dh = cfg.head_dim();
  int64_t M = cfg.mlp_hidden;
  int64_t V = cfg.vocab_size;
  int64_t L = cfg.n_layers;

  int64_t per_block = B*T*D       // ln1_out
                    + B*T*D       // ln2_out
                    + B*T         // ln1 mean
                    + B*T         // ln1 inv_std
                    + B*T         // ln2 mean
                    + B*T         // ln2 inv_std
                    + B*T*D       // attn_out
                    + B*T*M       // fc1_out
                    + B*T*M       // gelu_out
                    + B*T*D       // fc2_out
                    + B*T*3*D     // qkv
                    + B*H*T*Dh    // Q
                    + B*H*T*Dh    // K
                    + B*H*T*Dh    // V
                    + B*H*T*T     // scores
                    + B*H*T*T     // probs
                    + B*H*T*Dh    // context
                    + B*T*D;      // context_merged

  int64_t total = B*T*D           // embed_out
                + L * per_block
                + B*T*D           // final_ln_out
                + B*T             // final_ln mean
                + B*T             // final_ln inv_std
                + B*T*V;          // logits

  float* buf = nullptr;
  cudaMalloc(&buf, total * sizeof(float));
  cudaMemset(buf, 0, total * sizeof(float));

  float* ptr = buf;
  auto advance = [&](int64_t n) -> float* {
    float* p = ptr;
    ptr += n;
    return p;
  };

  fwd.embed_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};

  fwd.blocks.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    auto& blk = fwd.blocks[l];
    blk.ln1_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.ln2_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.ln1_state.mean = {advance(B*T), {B*T}, {1}};
    blk.ln1_state.inv_std = {advance(B*T), {B*T}, {1}};
    blk.ln2_state.mean = {advance(B*T), {B*T}, {1}};
    blk.ln2_state.inv_std = {advance(B*T), {B*T}, {1}};
    blk.attn_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.fc1_out = {advance(B*T*M), {B,T,M}, {T*M, M, 1}};
    blk.gelu_out = {advance(B*T*M), {B,T,M}, {T*M, M, 1}};
    blk.fc2_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.attn_state.qkv = {advance(B*T*3*D), {B,T,3*D}, {T*3*D, 3*D, 1}};
    blk.attn_state.Q = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.K = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.V = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.scores = {advance(B*H*T*T), {B,H,T,T}, {H*T*T, T*T, T, 1}};
    blk.attn_state.probs = {advance(B*H*T*T), {B,H,T,T}, {H*T*T, T*T, T, 1}};
    blk.attn_state.context = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.context_merged = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
  }

  fwd.final_ln_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
  fwd.final_ln_state.mean = {advance(B*T), {B*T}, {1}};
  fwd.final_ln_state.inv_std = {advance(B*T), {B*T}, {1}};
  fwd.logits = {advance(B*T*V), {B*T, V}, {V, 1}};

  return buf;
}

// Allocate params and wire views from a flat buffer.
float* allocate_params(const GPTConfig& cfg, GPTParams& params) {
  int64_t D = cfg.d_model;
  int64_t V = cfg.vocab_size;
  int64_t S = cfg.max_seq_len;
  int64_t M = cfg.mlp_hidden;
  int64_t L = cfg.n_layers;

  int64_t count = cfg.approx_param_count();
  float* buf = nullptr;
  cudaMalloc(&buf, count * sizeof(float));

  float* ptr = buf;
  auto advance = [&](int64_t n) -> float* {
    float* p = ptr;
    ptr += n;
    return p;
  };

  params.token_embedding = {advance(V*D), {V, D}, {D, 1}};
  params.position_embedding = {advance(S*D), {S, D}, {D, 1}};

  params.layers.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    auto& lp = params.layers[l];
    lp.ln1.gamma = {advance(D), {D}, {1}};
    lp.ln1.beta = {advance(D), {D}, {1}};
    lp.ln1.eps = cfg.layernorm_eps;
    lp.attn.W_qkv = {advance(3*D*D), {3*D, D}, {D, 1}};
    if (cfg.use_bias) {
      lp.attn.b_qkv = {advance(3*D), {3*D}, {1}};
    } else {
      lp.attn.b_qkv = {nullptr, {3*D}, {1}};
    }
    lp.attn.W_o = {advance(D*D), {D, D}, {D, 1}};
    if (cfg.use_bias) {
      lp.attn.b_o = {advance(D), {D}, {1}};
    } else {
      lp.attn.b_o = {nullptr, {D}, {1}};
    }
    lp.ln2.gamma = {advance(D), {D}, {1}};
    lp.ln2.beta = {advance(D), {D}, {1}};
    lp.ln2.eps = cfg.layernorm_eps;
    lp.fc1.weight = {advance(M*D), {M, D}, {D, 1}};
    if (cfg.use_bias) {
      lp.fc1.bias = {advance(M), {M}, {1}};
    } else {
      lp.fc1.bias = {nullptr, {M}, {1}};
    }
    lp.fc1.use_bias = cfg.use_bias;
    lp.fc2.weight = {advance(D*M), {D, M}, {M, 1}};
    if (cfg.use_bias) {
      lp.fc2.bias = {advance(D), {D}, {1}};
    } else {
      lp.fc2.bias = {nullptr, {D}, {1}};
    }
    lp.fc2.use_bias = cfg.use_bias;
  }

  params.final_ln.gamma = {advance(D), {D}, {1}};
  params.final_ln.beta = {advance(D), {D}, {1}};
  params.final_ln.eps = cfg.layernorm_eps;

  if (cfg.tie_embeddings) {
    params.lm_head = params.token_embedding;
  } else {
    params.lm_head = {advance(V*D), {V, D}, {D, 1}};
  }

  return buf;
}

}  // namespace

class GPTCheckpointRoundtripTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();

    // Create a unique temp directory for checkpoints.
    ckpt_dir_ = "/tmp/gpt_ckpt_test_" + std::to_string(getpid());
  }

  void TearDown() override {
    // Clean up checkpoint directory.
    std::filesystem::remove_all(ckpt_dir_);
  }

  CudaStream stream_;
  std::string ckpt_dir_;
};

TEST_F(GPTCheckpointRoundtripTest, SaveLoadProducesIdenticalOutput) {
  // Tiny config.
  GPTConfig cfg;
  cfg.vocab_size   = 50;
  cfg.max_seq_len  = 16;
  cfg.n_layers     = 1;
  cfg.n_heads      = 2;
  cfg.d_model      = 16;
  cfg.mlp_hidden   = 64;
  cfg.tie_embeddings = false;
  cfg.use_bias     = true;
  cfg.layernorm_eps = 1e-5f;

  const int64_t B = 1, T = 4;
  int64_t param_count = cfg.approx_param_count();

  // ---- Original model ----
  GPTParams params1;
  float* param_buf1 = allocate_params(cfg, params1);
  ASSERT_NE(param_buf1, nullptr);

  // Initialize with random weights.
  std::vector<float> h_params(param_count);
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  for (auto& v : h_params) v = dist(rng);
  cudaMemcpy(param_buf1, h_params.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);

  // Fixed input.
  std::vector<int32_t> h_ids = {1, 5, 10, 20};  // T=4 tokens
  int32_t* d_ids = nullptr;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};

  // Forward pass 1.
  GPTForwardState fwd1;
  float* fwd_buf1 = allocate_forward_state(cfg, B, T, fwd1);
  auto status1 = gpt_forward(input_ids, cfg, params1, fwd1, stream_);
  ASSERT_TRUE(status1.ok()) << status1.message();
  stream_.synchronize();

  // Copy logits to host.
  int64_t logits_size = B * T * cfg.vocab_size;
  std::vector<float> logits1(logits_size);
  cudaMemcpy(logits1.data(), fwd1.logits.data,
             logits_size * sizeof(float), cudaMemcpyDeviceToHost);

  // ---- Save checkpoint ----
  // save_checkpoint needs opt_m and opt_v buffers.
  float* dummy_opt_m = nullptr;
  float* dummy_opt_v = nullptr;
  cudaMalloc(&dummy_opt_m, param_count * sizeof(float));
  cudaMalloc(&dummy_opt_v, param_count * sizeof(float));
  cudaMemset(dummy_opt_m, 0, param_count * sizeof(float));
  cudaMemset(dummy_opt_v, 0, param_count * sizeof(float));

  checkpoint::CheckpointMetadata meta;
  meta.step = 42;
  meta.param_count = param_count;
  meta.config_json = "{}";

  auto save_status = checkpoint::save_checkpoint(
      ckpt_dir_, param_buf1, dummy_opt_m, dummy_opt_v, param_count, meta);
  ASSERT_TRUE(save_status.ok()) << save_status.message();

  // ---- Load checkpoint into fresh buffers ----
  GPTParams params2;
  float* param_buf2 = allocate_params(cfg, params2);
  ASSERT_NE(param_buf2, nullptr);

  // Zero it out first to prove loading actually overwrites.
  cudaMemset(param_buf2, 0, param_count * sizeof(float));

  float* loaded_opt_m = nullptr;
  float* loaded_opt_v = nullptr;
  cudaMalloc(&loaded_opt_m, param_count * sizeof(float));
  cudaMalloc(&loaded_opt_v, param_count * sizeof(float));

  checkpoint::CheckpointMetadata loaded_meta;
  auto load_status = checkpoint::load_checkpoint(
      ckpt_dir_, param_buf2, loaded_opt_m, loaded_opt_v,
      param_count, loaded_meta);
  ASSERT_TRUE(load_status.ok()) << load_status.message();

  // ---- Forward pass 2 with loaded params ----
  GPTForwardState fwd2;
  float* fwd_buf2 = allocate_forward_state(cfg, B, T, fwd2);
  auto status2 = gpt_forward(input_ids, cfg, params2, fwd2, stream_);
  ASSERT_TRUE(status2.ok()) << status2.message();
  stream_.synchronize();

  // Copy logits to host.
  std::vector<float> logits2(logits_size);
  cudaMemcpy(logits2.data(), fwd2.logits.data,
             logits_size * sizeof(float), cudaMemcpyDeviceToHost);

  // ---- Verify identical outputs ----
  for (int64_t i = 0; i < logits_size; ++i) {
    EXPECT_EQ(logits1[i], logits2[i])
        << "Logits differ at index " << i
        << ": original=" << logits1[i]
        << " loaded=" << logits2[i];
  }

  // Cleanup.
  cudaFree(loaded_opt_v);
  cudaFree(loaded_opt_m);
  cudaFree(fwd_buf2);
  cudaFree(param_buf2);
  cudaFree(dummy_opt_v);
  cudaFree(dummy_opt_m);
  cudaFree(fwd_buf1);
  cudaFree(d_ids);
  cudaFree(param_buf1);
}
