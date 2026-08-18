# Backward Scratch Plan Spec

## Problem Statement

The fork trainer now has a shared device scratch arena, but the training runtime still
knows the model backward allocation formula. That makes the trainer depend on the
implementation details of model, block, and attention backward passes. When any of
those modules adds, removes, or resizes a scratch tensor, the trainer must be updated
by hand or a training step can fail at runtime with an exhausted arena.

## Solution

Move backward scratch sizing behind a model-owned module interface. The trainer should
ask the model layer for the scratch requirement for a given configuration and batch
shape, reserve that many bytes in the arena, and then run backward without carrying
the allocation formula itself.

The module should express the planning result in bytes, validate the same shape facts
that model backward requires, and account for arena alignment padding. The first
implementation may preserve the current allocation counts and tensor counts exactly;
the important change is locality, not a new memory optimization.

## User Stories

1. As a trainer maintainer, I want the training runtime to reserve backward scratch
   through a model-owned interface, so that trainer changes do not need to mirror
   model internals.
2. As a model maintainer, I want scratch sizing to live beside the backward allocation
   sequence, so that adding a backward temporary has one obvious planning update.
3. As a test author, I want host-testable scratch planning, so that size regressions
   can fail without launching CUDA kernels.
4. As a runtime user, I want under-sized scratch plans to fail loudly in focused tests,
   so that training steps do not discover stale formulas late.
5. As a future attention maintainer, I want attention backward scratch needs to be
   independently nameable, so that attention changes do not leak into trainer math.

## Implementation Decisions

- Add a model-owned backward scratch planning module whose interface returns the
  number of bytes required for a full model backward pass for one batch shape.
- Keep `DeviceScratchArena` as the allocation module. The new module plans required
  capacity; it does not allocate or own device memory.
- Put shape validation in the planning module and keep error modes aligned with model
  backward validation.
- Account for `DeviceScratchArena` alignment padding in the returned byte count.
- Let model, block, and attention scratch subplans exist behind the model interface
  only if they improve locality. The trainer should see one model-level answer.
- Preserve current behavior and memory requirements unless tests expose an existing
  sizing bug.

## Testing Decisions

- Add host tests for invalid model shapes, zero batch dimensions, and the exact current
  scratch byte requirement for a small multi-layer configuration.
- Add a CUDA regression test that reserves exactly the model-owned planned capacity
  and runs a trainer step or model backward path successfully.
- Keep existing trainer, model backward, block backward, and attention backward tests
  as behavioral coverage.
- The preferred test seam is the new planning interface plus the existing trainer step.

## Out of Scope

- Reducing scratch memory usage.
- Reordering model backward allocations.
- Changing `DeviceScratchArena` allocation behavior.
- Changing optimizer, DDP, or checkpoint memory ownership.
- Introducing a second scratch allocator or hidden per-module `cudaMalloc`.

## Further Notes

This follows the architecture report top recommendation in
`/tmp/architecture-review-20260818T020253Z.html`: make backward scratch planning a
model module so trainer logic stops duplicating model, block, and attention internals.
