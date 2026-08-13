#!/bin/bash
# Loud local-run verifier for the fork's hermetic CUDA infra.
#
# Walks a fixed acceptance ladder on the local GPUs and fails LOUDLY with a
# distinct exit code per rung, so a failure names exactly which layer broke
# (modeled on the monarch Local Run ladder). This proves the *infrastructure*
# — rootfs + Bazel + hermetic CUDA toolchain + a GPU that runs a CUTLASS kernel
# — not the //src fork trainer, whose kernels are out of scope until infra
# lands (ADR 0005, CONTEXT.md).
#
# Ladder (exit codes):
#   21  preflight  : rootfs marker, bazel/nvcc present, exactly one visible GPU
#   22  build      : `bazel build --config=cuda` of the smoke target
#   23  gemm smoke : single-GPU CUTLASS GEMM runs and matches a reference kernel
#   24  gtest      : (optional, --with-gtest) //src gpu-tagged unit tests
#    0  all requested rungs passed
#
# Usage:
#   scripts/run_local_gpu_smoke.sh [--with-gtest] [-- <extra bazel args>]
#
# Re-execs itself into the hermetic bwrap rootfs unless already inside it
# (guarded by CUTLASS_IN_ROOTFS). Honors a caller-set CUDA_VISIBLE_DEVICES that
# names exactly one device; unset defaults to device 0; empty is a hard error.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok() { printf '\033[1;32m%s\033[0m\n' "$*"; }
die() {
  local code="$1"; shift
  printf '\033[1;31mFAIL[%s]: %s\033[0m\n' "$code" "$*" >&2
  exit "$code"
}

WITH_GTEST=0
BAZEL_EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-gtest) WITH_GTEST=1; shift ;;
    --in-rootfs) shift ;;  # informational; the marker is the real guard
    --) shift; BAZEL_EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

HOST_PYTHON="$(command -v python3 || command -v python || true)"
[[ -n "$HOST_PYTHON" ]] || die 21 "python3/python required to validate CUDA_VISIBLE_DEVICES"

# Resolve the single-GPU selection before deciding whether to enter the rootfs,
# so the value is validated once and passed through consistently.
log "preflight: GPU visibility"
visibility_out="$("$HOST_PYTHON" "$REPO_ROOT/scripts/local_gpu_smoke.py" cuda-visible-devices)" \
  || die 21 "invalid CUDA_VISIBLE_DEVICES for a single-GPU smoke"
mapfile -t visibility <<< "$visibility_out"
CUDA_VISIBLE_DEVICES="${visibility[0]}"
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (${visibility[1]})"

# Re-exec into the hermetic rootfs unless already inside it.
if [[ "${CUTLASS_IN_ROOTFS:-0}" != "1" ]]; then
  log "preflight: entering hermetic bwrap rootfs"
  gtest_arg=(); [[ "$WITH_GTEST" -eq 1 ]] && gtest_arg=(--with-gtest)
  exec "$REPO_ROOT/scripts/rootfs/enter_rootfs.sh" -- \
    scripts/run_local_gpu_smoke.sh "${gtest_arg[@]}" \
    ${BAZEL_EXTRA:+-- "${BAZEL_EXTRA[@]}"}
fi

echo "inside hermetic bwrap rootfs"
command -v bazel >/dev/null || die 21 "bazel not found in rootfs"
command -v nvcc  >/dev/null || die 21 "nvcc not found in rootfs (CUDA_HOME=${CUDA_HOME:-unset})"
[[ -n "${CUDA_HOME:-}" ]] || die 21 "CUDA_HOME is unset inside the rootfs"
echo "bazel=$(command -v bazel)  nvcc=$(command -v nvcc)  CUDA_HOME=$CUDA_HOME"

# nvidia-smi is bound in read-only; use it to confirm the driver sees a GPU. It
# is not strictly required (the smoke's cudaGetDeviceProperties is the real
# gate) but gives a loud, early, human-readable preflight signal.
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -L || die 21 "nvidia-smi could not list any GPU"
fi
ok "preflight: PASS"

SMOKE_TGT="//tests/smoke:cutlass_gemm_smoke"

log "build: bazel build --config=cuda $SMOKE_TGT"
# We are past the CUTLASS_IN_ROOTFS guard above, so the build-level rootfs guard
# (//tools/bazel:rootfs_guard, ADR 0006) will pass here. If this script is ever
# invoked with the marker set but outside a real rootfs, that guard is the
# belt-and-suspenders backstop that still fails the CUDA build loudly.
bazel build --config=cuda "${BAZEL_EXTRA[@]}" "$SMOKE_TGT" \
  || die 22 "hermetic CUDA build of the GEMM smoke failed"
ok "build: PASS"

log "gemm smoke: single-GPU CUTLASS GEMM"
SMOKE_BIN="$REPO_ROOT/$(bazel cquery --config=cuda --output=files "$SMOKE_TGT" 2>/dev/null | head -1)"
[[ -x "$SMOKE_BIN" ]] || die 23 "could not locate built smoke binary at $SMOKE_BIN"
"$SMOKE_BIN" || die 23 "single-GPU CUTLASS GEMM smoke failed (see stderr above)"
ok "gemm smoke: PASS"

if [[ "$WITH_GTEST" -eq 1 ]]; then
  log "gtest: //src gpu-tagged unit tests"
  # The //src scaffold kernels are out of scope; run their gpu tests only when
  # explicitly requested, and treat a failure as its own rung so infra proof is
  # never blocked by feature-incomplete scaffold tests.
  bazel test --config=cuda --test_tag_filters=gpu "${BAZEL_EXTRA[@]}" //src/... \
    || die 24 "//src gpu-tagged gtest run failed (scaffold feature work, out of infra scope)"
  ok "gtest: PASS"
fi

log "summary"
ok "preflight  : PASS"
ok "build      : PASS"
ok "gemm smoke : PASS"
[[ "$WITH_GTEST" -eq 1 ]] && ok "gtest      : PASS"
ok "local GPU infra verified"
exit 0
