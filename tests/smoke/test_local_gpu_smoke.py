#!/usr/bin/env python3
"""Unit tests for scripts/local_gpu_smoke.py (host-runnable, no GPU needed).

Run with: python3 -m pytest tests/smoke/test_local_gpu_smoke.py
      or:  python3 tests/smoke/test_local_gpu_smoke.py  (falls back to a
           minimal runner if pytest is unavailable in the rootfs).
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is not a package; import the module by path.
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import local_gpu_smoke as smoke  # noqa: E402


def test_unset_defaults_to_device_zero():
    value, defaulted = smoke.validated_cuda_visible_devices(None)
    assert value == "0"
    assert defaulted is True


def test_single_device_is_honored():
    value, defaulted = smoke.validated_cuda_visible_devices("3")
    assert value == "3"
    assert defaulted is False


def test_empty_string_is_hard_error():
    try:
        smoke.validated_cuda_visible_devices("")
    except smoke.CudaVisibleDevicesError:
        pass
    else:
        raise AssertionError("empty CUDA_VISIBLE_DEVICES must raise")


def test_multiple_devices_rejected():
    try:
        smoke.validated_cuda_visible_devices("0,1")
    except smoke.CudaVisibleDevicesError:
        pass
    else:
        raise AssertionError("multi-device CUDA_VISIBLE_DEVICES must raise")


def test_trailing_comma_rejected():
    try:
        smoke.validated_cuda_visible_devices("0,")
    except smoke.CudaVisibleDevicesError:
        pass
    else:
        raise AssertionError("malformed CUDA_VISIBLE_DEVICES must raise")


def _run_without_pytest() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_without_pytest())
