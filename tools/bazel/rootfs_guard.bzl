"""Build-level guard enforcing that CUDA/GPU work runs inside the bwrap rootfs.

The fork's isolation model (ADR 0002/0006) requires that any CUDA compile/link
happens inside the hermetic bwrap rootfs, where the one pinned CUDA 12.8 devel
toolchain lives. This rule makes that invariant a *build* invariant rather than
a convention: it runs a check action that reads CUTLASS_IN_ROOTFS at execution
time and hard-fails LOUDLY if the build is happening on the host.

It is wired as a dependency of //third_party/cuda:cudart, which every CUDA
target in the repo depends on, so the guard fires for the whole `--config=cuda`
build and cannot be skipped by targeting a specific label. Because those targets
only build under `--config=cuda`, the guard never touches the CPU header-only
subset.

Escape hatch: set CUTLASS_ALLOW_HOST_CUDA=1 to intentionally build CUDA on the
host (e.g. an expert debugging the toolchain outside the sandbox). This is a
deliberate, visible override — the default is to fail.

Both env vars must be passed through with --action_env (see .bazelrc,
build:cuda) so the action sees them.
"""

def _rootfs_guard_impl(ctx):
    # Emit a trivial .cc so the guard can live in a cc_library's srcs: compiling
    # (and thus generating) it forces the check action to run whenever any CUDA
    # target that depends on the wrapper is built. The file content is fixed; its
    # generation is what's gated on the guard passing.
    generated = ctx.actions.declare_file(ctx.label.name + "_stamp.cc")
    ctx.actions.run_shell(
        outputs = [generated],
        mnemonic = "RootfsGuard",
        progress_message = "Enforcing bwrap rootfs for CUDA build (%s)" % ctx.label,
        # Keep the guard honest: no remote cache hit should let a host build
        # reuse a stamp generated inside the rootfs. (Local disk cache is still
        # keyed on the action_env values, so a host vs rootfs build differ.)
        execution_requirements = {"no-remote-cache": "1"},
        command = """
set -eu
if [ "${CUTLASS_IN_ROOTFS:-0}" = "1" ]; then
  echo "// generated inside bwrap rootfs" > "$1"
  exit 0
fi
if [ "${CUTLASS_ALLOW_HOST_CUDA:-0}" = "1" ]; then
  echo "// generated on host (escape hatch CUTLASS_ALLOW_HOST_CUDA=1)" > "$1"
  exit 0
fi
cat >&2 <<'MSG'

  ============================================================================
  BLOCKED: CUDA/GPU build attempted OUTSIDE the hermetic bwrap rootfs.

  This fork requires CUDA work to run inside the rootfs, which carries the one
  pinned CUDA 12.8 devel toolchain (nvcc + host C++). Building CUDA on the bare
  host has no nvcc and no version-matched driver, and is not supported.

  Do this instead:
      scripts/run_local_gpu_smoke.sh            # build + verify on a local GPU
      scripts/rootfs/enter_rootfs.sh -- bash    # interactive shell in the rootfs

  Expert escape hatch (you are on your own):
      CUTLASS_ALLOW_HOST_CUDA=1 bazel build --config=cuda //...
  ============================================================================

MSG
exit 1
""",
        arguments = [generated.path],
        # --action_env values (CUTLASS_IN_ROOTFS / CUTLASS_ALLOW_HOST_CUDA, set
        # under build:cuda) are injected only into the default shell env, so the
        # action must opt in to see them.
        use_default_shell_env = True,
    )
    return [DefaultInfo(files = depset([generated]))]

rootfs_guard = rule(
    implementation = _rootfs_guard_impl,
    doc = "Generates a stamp .cc only if the CUDA build runs inside the rootfs.",
)
