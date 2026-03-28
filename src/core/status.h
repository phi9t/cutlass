#pragma once

// Layer 1 — Core runtime: status / error utilities.
//
// Provides a lightweight Result<T> type and CudaStatus wrapper so that
// every layer above can propagate errors without exceptions.

#include <cstdint>
#include <string>
#include <variant>

namespace gpt {

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

enum class StatusCode : uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kOutOfMemory,
  kCudaError,
  kNcclError,
  kInternalError,
  kNotImplemented,
};

// ---------------------------------------------------------------------------
// Status — lightweight error carrier
// ---------------------------------------------------------------------------

class Status {
 public:
  Status() : code_(StatusCode::kSuccess) {}
  explicit Status(StatusCode code) : code_(code) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status{}; }

  [[nodiscard]] bool ok() const { return code_ == StatusCode::kSuccess; }
  [[nodiscard]] StatusCode code() const { return code_; }
  [[nodiscard]] const std::string& message() const { return message_; }

 private:
  StatusCode code_;
  std::string message_;
};

// ---------------------------------------------------------------------------
// Result<T> — value-or-error carrier
// ---------------------------------------------------------------------------

template <typename T>
class Result {
 public:
  /* implicit */ Result(T value) : data_(std::move(value)) {}
  /* implicit */ Result(Status status) : data_(std::move(status)) {}

  [[nodiscard]] bool ok() const { return std::holds_alternative<T>(data_); }

  [[nodiscard]] const T& value() const { return std::get<T>(data_); }
  [[nodiscard]] T& value() { return std::get<T>(data_); }
  [[nodiscard]] T&& take() { return std::get<T>(std::move(data_)); }

  [[nodiscard]] const Status& status() const {
    return std::get<Status>(data_);
  }

 private:
  std::variant<T, Status> data_;
};

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------

#define GPT_RETURN_IF_ERROR(expr)   \
  do {                              \
    auto _status = (expr);          \
    if (!_status.ok()) return _status; \
  } while (false)

#define GPT_CHECK_CUDA(call)                                        \
  do {                                                              \
    cudaError_t _err = (call);                                      \
    if (_err != cudaSuccess) {                                      \
      return ::gpt::Status(::gpt::StatusCode::kCudaError,          \
                           cudaGetErrorString(_err));               \
    }                                                               \
  } while (false)

}  // namespace gpt
