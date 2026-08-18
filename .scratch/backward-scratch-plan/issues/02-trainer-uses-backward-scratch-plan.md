# 02 — Make trainer reserve scratch through the model plan

**What to build:** the training runtime asks the model-owned scratch planning module
for backward capacity before each step instead of carrying its own allocation formula.

**Blocked by:** 01 — Add a model backward scratch planning module.

**Status:** ready-for-agent

- [ ] The trainer no longer contains a model/block/attention scratch formula.
- [ ] Trainer tests pass with the same behavior as before.
- [ ] Invalid planning results surface as `Status` failures before backward starts.
- [ ] CUDA tests run inside the rootfs.
