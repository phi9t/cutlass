# ADR 0006 — a build-level guard makes the rootfs mandatory for CUDA work

- Status: Accepted
- Date: 2026-08-13
- Depends on: ADR 0002 (rootfs is the outer boundary), ADR 0003 (CUDA provenance).

## Context

ADR 0002 established the invariant: **CUDA compile/link runs inside the bwrap rootfs**,
which carries the one pinned CUDA 12.8 devel toolchain. Until now that invariant rested
on a *single convention* — the `run_in_rootfs.sh` / `run_local_gpu_smoke.sh` scripts check
`CUTLASS_IN_ROOTFS=1` and re-exec into the rootfs.

The gap: nothing stopped a human or a fresh agent from running
`bazel build --config=cuda //...` directly on the bare host. There is no `nvcc` on the
host, so such a build fails — but *confusingly* (a missing-compiler error deep in an
action), not with a message that says "you skipped the rootfs." The invariant was
enforced only at the edges (the verifier scripts), not at the tool everyone actually
reaches for (Bazel).

Options considered for closing the gap:
- (a) **Hard-fail loudly** at the Bazel level for any CUDA build outside the rootfs, with
  an explicit escape hatch.
- (b) **Auto re-exec** host CUDA builds into the rootfs via a `bazel` wrapper/shim.
- (c) **Warn only** — print a warning but let the host build proceed.

## Decision

**Adopt (a): a build-level guard that hard-fails loudly.** A `rootfs_guard` action
(`//tools/bazel:rootfs_guard`) generates a trivial stamp source only if the build is
inside the rootfs; otherwise it prints an actionable BLOCKED message (pointing at
`scripts/run_local_gpu_smoke.sh` and `scripts/rootfs/enter_rootfs.sh`) and exits non-zero,
failing the build.

- **Trigger:** the guard is a dependency of `//third_party/cuda:cudart`, which *every*
  CUDA target depends on, added via `select()` on a `cuda_enabled` `config_setting`
  (`@rules_cuda//cuda:enable`). So it fires for the whole `--config=cuda` build and never
  for the CPU header-only subset.
- **Signal:** the marker `CUTLASS_IN_ROOTFS=1` (set by `enter_rootfs.sh`) and the escape
  hatch `CUTLASS_ALLOW_HOST_CUDA=1` reach the action via `--action_env` under
  `build:cuda`; the action uses the default shell env so those values are visible.
- **Escape hatch:** `CUTLASS_ALLOW_HOST_CUDA=1` lets an expert intentionally build CUDA on
  the host. Deliberate and visible — the default is to fail.

Enforcement is layered at all three points: this build guard, the `CONTEXT.md` glossary +
`AGENTS.md` rule, and the existing verifier-script preflights.

## Consequences

- The rootfs invariant is now enforced by the **build system itself** and cannot be
  skipped by targeting a specific label — the strongest, least-bypassable point.
- A host CUDA build now fails *early and legibly* with the fix in the message, instead of
  a deep nvcc-not-found error.
- Rejected (b) auto re-exec: convenient but hides what's happening, is harder to reason
  about, and can surprise CI (a build silently changing environments). Rejected (c) warn
  only: does not actually enforce the invariant.
- Cost: `//third_party/cuda:cudart` is a thin `cc_library` wrapper rather than a bare
  `alias`, and the guard adds one tiny action to every CUDA build. Negligible, and the
  local action cache keys it on the forwarded env so host vs rootfs never share a result.
