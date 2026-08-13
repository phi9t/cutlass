#!/usr/bin/env python3
"""Testable helpers for the local single-GPU CUTLASS smoke verifier.

The shell entrypoint (scripts/run_local_gpu_smoke.sh) stays thin: it re-execs
into the bwrap rootfs and walks the acceptance ladder. The risky parsing lives
here so it can be unit-tested on the host without a GPU (see
tests/smoke/test_local_gpu_smoke.py). Mirrors the monarch pattern of moving
verifier logic into a focused, importable Python module.

Subcommands:
  cuda-visible-devices   Resolve/validate CUDA_VISIBLE_DEVICES for a 1-GPU run.
                         Prints two lines: the value to export, then
                         "defaulted" or "honored".
"""

from __future__ import annotations

import argparse
import os
import sys

# The smoke pins itself to a single GPU. When the caller has not chosen one we
# default to device 0 rather than exposing all GPUs, so the smoke is a true
# single-GPU proof and does not contend with other work on the box.
DEFAULT_CUDA_VISIBLE_DEVICES = "0"


class CudaVisibleDevicesError(ValueError):
    """Raised when CUDA_VISIBLE_DEVICES is set but not a single valid device."""


def validated_cuda_visible_devices(value: str | None) -> tuple[str, bool]:
    """Resolve the single-GPU device selection.

    Returns (value_to_export, defaulted). An unset value defaults to device 0.
    An explicitly empty value is a hard error (it would hide all GPUs, which the
    preflight must treat as a failure, not silently widen). A set value must
    name exactly one non-empty device.
    """
    if value is None:
        return DEFAULT_CUDA_VISIBLE_DEVICES, True

    if value == "":
        raise CudaVisibleDevicesError(
            "CUDA_VISIBLE_DEVICES is empty, which hides all GPUs; unset it to "
            "default to device 0 or name exactly one device"
        )

    devices = value.split(",")
    if len(devices) != 1 or devices[0] == "":
        raise CudaVisibleDevicesError(
            "the single-GPU smoke needs exactly one device in "
            f"CUDA_VISIBLE_DEVICES, got: {value!r}"
        )
    return value, False


def _cmd_cuda_visible_devices(args: argparse.Namespace) -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if args.value is None else args.value
    # argparse cannot distinguish "flag absent" from "" cleanly, so treat the
    # sentinel None (no --value) as "read the environment".
    try:
        value, defaulted = validated_cuda_visible_devices(raw)
    except CudaVisibleDevicesError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(value)
    print("defaulted" if defaulted else "honored")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "cuda-visible-devices",
        help="resolve/validate CUDA_VISIBLE_DEVICES for a single-GPU run",
    )
    # --value is mainly for tests; production callers rely on the environment.
    p.add_argument("--value", default=None)
    p.set_defaults(func=_cmd_cuda_visible_devices)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
