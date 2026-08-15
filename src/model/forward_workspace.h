#pragma once

// Layer 5 — Model: persistent forward workspace.

#include <cstdint>
#include <vector>

#include "src/core/status.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_model.h"

namespace gpt {
namespace model {

class ForwardWorkspace {
 public:
  ForwardWorkspace() = default;

  Status ensure(const GPTConfig& config, int64_t B, int64_t T);

  [[nodiscard]] GPTForwardState& state() { return state_; }
  [[nodiscard]] const GPTForwardState& state() const { return state_; }

  void release();
  ~ForwardWorkspace();

  ForwardWorkspace(const ForwardWorkspace&) = delete;
  ForwardWorkspace& operator=(const ForwardWorkspace&) = delete;

 private:
  GPTForwardState state_;
  std::vector<float*> buffers_;
  std::vector<attention::AttentionWorkspace> attention_workspaces_;
  int64_t capacity_B_ = 0;
  int64_t capacity_T_ = 0;
};

}  // namespace model
}  // namespace gpt
