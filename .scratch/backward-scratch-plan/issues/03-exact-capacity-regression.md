# 03 — Add exact-capacity backward scratch regression coverage

**What to build:** a CUDA regression that reserves exactly the model-owned planned
scratch capacity and proves the relevant backward path completes without exhausting
`DeviceScratchArena`.

**Blocked by:** 02 — Make trainer reserve scratch through the model plan.

**Status:** ready-for-agent

- [ ] The test fails if the plan underestimates scratch bytes.
- [ ] The test uses the same rootfs CUDA path as the existing trainer tests.
- [ ] Existing model, block, attention, and trainer tests remain green.
- [ ] No hidden per-call `cudaMalloc` is introduced for backward scratch.
