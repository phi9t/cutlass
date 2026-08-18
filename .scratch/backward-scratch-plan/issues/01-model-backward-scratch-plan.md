# 01 — Add a model backward scratch planning module

**What to build:** a model-owned planning interface that returns the required
`DeviceScratchArena` capacity for a full model backward pass for one configuration
and batch shape.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The interface validates empty batch dimensions and invalid model shapes.
- [ ] The returned capacity accounts for arena alignment padding.
- [ ] Host tests cover the exact current small-configuration byte count.
- [ ] The trainer does not call the new interface yet.
