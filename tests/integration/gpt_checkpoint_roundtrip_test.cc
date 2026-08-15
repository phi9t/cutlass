// Tier C — Integration: checkpoint save/load produces identical model behavior.

#include "gtest/gtest.h"
#include "src/checkpoint/checkpoint.h"
#include "src/model/gpt_model.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace gpt;

namespace {

struct ZeroLayerForwardBuffers {
  float* embed = nullptr;
  float* mean = nullptr;
  float* inv_std = nullptr;
  float* final = nullptr;
  float* logits = nullptr;

  model::GPTForwardState make_state(int64_t B, int64_t T, int64_t D,
                                    int64_t vocab) {
    model::GPTForwardState state;
    state.embed_out = Tensor3D<float>{embed, {B, T, D}, {T * D, D, 1}};
    state.final_ln_out = Tensor3D<float>{final, {B, T, D}, {T * D, D, 1}};
    state.final_ln_state.mean = Tensor1D<float>{mean, {B * T}, {1}};
    state.final_ln_state.inv_std = Tensor1D<float>{inv_std, {B * T}, {1}};
    state.logits = Tensor2D<float>{logits, {B * T, vocab}, {vocab, 1}};
    return state;
  }

  void allocate(int64_t B, int64_t T, int64_t D, int64_t vocab) {
    cudaMalloc(&embed, B * T * D * sizeof(float));
    cudaMalloc(&mean, B * T * sizeof(float));
    cudaMalloc(&inv_std, B * T * sizeof(float));
    cudaMalloc(&final, B * T * D * sizeof(float));
    cudaMalloc(&logits, B * T * vocab * sizeof(float));
  }

  void release() {
    cudaFree(embed);
    cudaFree(mean);
    cudaFree(inv_std);
    cudaFree(final);
    cudaFree(logits);
    embed = nullptr;
    mean = nullptr;
    inv_std = nullptr;
    final = nullptr;
    logits = nullptr;
  }
};

}  // namespace

TEST(GPTCheckpointRoundtripTest, LoadProducesIdenticalForwardLogits) {
  const std::string dir = "/tmp/gpt_checkpoint_roundtrip_integration";
  std::filesystem::remove_all(dir);

  const int64_t B = 1;
  const int64_t T = 3;
  const int64_t D = 4;
  const int64_t vocab = 8;

  model::GPTConfig config;
  config.vocab_size = vocab;
  config.max_seq_len = T;
  config.n_layers = 0;
  config.n_heads = 2;
  config.d_model = D;
  config.mlp_hidden = 8;
  config.use_bias = false;
  config.tie_embeddings = false;
  config.layernorm_eps = 1e-5f;

  const int64_t param_count = config.approx_param_count();
  std::vector<float> h_params(param_count);
  std::vector<float> h_m(param_count);
  std::vector<float> h_v(param_count);
  test::SimpleRng rng(123);
  rng.fill(h_params, 0.2f);
  rng.fill(h_m, 0.01f);
  rng.fill(h_v, 0.001f);

  float* d_params = nullptr;
  float* d_m = nullptr;
  float* d_v = nullptr;
  float* d_loaded_params = nullptr;
  float* d_loaded_m = nullptr;
  float* d_loaded_v = nullptr;
  int32_t* d_ids = nullptr;
  cudaMalloc(&d_params, param_count * sizeof(float));
  cudaMalloc(&d_m, param_count * sizeof(float));
  cudaMalloc(&d_v, param_count * sizeof(float));
  cudaMalloc(&d_loaded_params, param_count * sizeof(float));
  cudaMalloc(&d_loaded_m, param_count * sizeof(float));
  cudaMalloc(&d_loaded_v, param_count * sizeof(float));
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMemcpy(d_params, h_params.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, h_m.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_loaded_params, 0, param_count * sizeof(float));
  cudaMemset(d_loaded_m, 0, param_count * sizeof(float));
  cudaMemset(d_loaded_v, 0, param_count * sizeof(float));

  std::vector<int32_t> h_ids = {1, 3, 5};
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  checkpoint::CheckpointMetadata meta;
  meta.step = 17;
  meta.param_count = param_count;
  meta.dataset_cursor = 96;
  meta.config_json = "{\"test\":\"gpt_checkpoint_roundtrip\"}";

  auto status = checkpoint::save_checkpoint(
      dir, d_params, d_m, d_v, param_count, meta);
  ASSERT_TRUE(status.ok()) << status.message();

  checkpoint::CheckpointMetadata loaded_meta;
  status = checkpoint::load_checkpoint(
      dir, d_loaded_params, d_loaded_m, d_loaded_v, param_count, loaded_meta);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(loaded_meta.step, meta.step);
  EXPECT_EQ(loaded_meta.param_count, meta.param_count);
  EXPECT_EQ(loaded_meta.dataset_cursor, meta.dataset_cursor);
  EXPECT_EQ(loaded_meta.config_json, meta.config_json);

  auto stream_result = CudaStream::Create();
  ASSERT_TRUE(stream_result.ok());
  CudaStream stream = stream_result.take();

  model::GPTParams params;
  model::GPTParams loaded_params;
  ASSERT_TRUE(model::init_gpt_params(config, d_params, params).ok());
  ASSERT_TRUE(model::init_gpt_params(config, d_loaded_params, loaded_params).ok());

  ZeroLayerForwardBuffers original_buffers;
  ZeroLayerForwardBuffers loaded_buffers;
  original_buffers.allocate(B, T, D, vocab);
  loaded_buffers.allocate(B, T, D, vocab);
  model::GPTForwardState original_state =
      original_buffers.make_state(B, T, D, vocab);
  model::GPTForwardState loaded_state =
      loaded_buffers.make_state(B, T, D, vocab);

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  status = model::gpt_forward(input_ids, config, params, original_state, stream);
  ASSERT_TRUE(status.ok()) << status.message();
  status = model::gpt_forward(
      input_ids, config, loaded_params, loaded_state, stream);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(stream.synchronize().ok());

  std::vector<float> h_original_logits(B * T * vocab);
  std::vector<float> h_loaded_logits(B * T * vocab);
  cudaMemcpy(h_original_logits.data(), original_state.logits.data,
             h_original_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_loaded_logits.data(), loaded_state.logits.data,
             h_loaded_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_TRUE(test::vectors_near(h_original_logits, h_loaded_logits, 0.0f, 0.0f));

  original_buffers.release();
  loaded_buffers.release();
  cudaFree(d_params);
  cudaFree(d_m);
  cudaFree(d_v);
  cudaFree(d_loaded_params);
  cudaFree(d_loaded_m);
  cudaFree(d_loaded_v);
  cudaFree(d_ids);
  std::filesystem::remove_all(dir);
}
