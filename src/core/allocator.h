#pragma once

// Layer 1 — Core runtime: GPU memory allocators.
//
// Two allocator types:
//   PersistentArena — long-lived buffers (params, grads, optimizer state).
//   WorkspaceArena  — scratch space reset per step or per phase.

#include <cstddef>
#include <cstdint>

#include "src/core/status.h"

namespace gpt {

// ---------------------------------------------------------------------------
// PersistentArena — bump allocator for long-lived GPU buffers.
// ---------------------------------------------------------------------------

class PersistentArena {
 public:
  // Alignment for all allocations (256 bytes covers all CUDA requirements).
  static constexpr size_t kDefaultAlignment = 256;

  PersistentArena() = default;

  // Initialize the arena with `capacity_bytes` of device memory.
  Status init(size_t capacity_bytes);

  // Allocate `size` bytes, aligned to `alignment`.
  Result<void*> allocate(size_t size, size_t alignment = kDefaultAlignment);

  // Return total and used byte counts.
  [[nodiscard]] size_t capacity() const { return capacity_; }
  [[nodiscard]] size_t used() const { return offset_; }
  [[nodiscard]] size_t remaining() const { return capacity_ - offset_; }

  // Free the entire backing buffer.
  void release();

  ~PersistentArena();

  PersistentArena(const PersistentArena&) = delete;
  PersistentArena& operator=(const PersistentArena&) = delete;

 private:
  void* base_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;
};

// ---------------------------------------------------------------------------
// WorkspaceArena — resettable scratch allocator.
// ---------------------------------------------------------------------------

class WorkspaceArena {
 public:
  static constexpr size_t kDefaultAlignment = 256;

  WorkspaceArena() = default;

  Status init(size_t capacity_bytes);

  Result<void*> allocate(size_t size, size_t alignment = kDefaultAlignment);

  // Reset the arena to empty without freeing the backing buffer.
  void reset() { offset_ = 0; }

  [[nodiscard]] size_t capacity() const { return capacity_; }
  [[nodiscard]] size_t used() const { return offset_; }
  [[nodiscard]] size_t remaining() const { return capacity_ - offset_; }

  void release();

  ~WorkspaceArena();

  WorkspaceArena(const WorkspaceArena&) = delete;
  WorkspaceArena& operator=(const WorkspaceArena&) = delete;

 private:
  void* base_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;
};

}  // namespace gpt
