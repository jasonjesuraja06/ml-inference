"""
What this CPU actually supports.

Dynamic INT8 quantization is a bet on integer dot-product instructions, and
which ones a machine has decides both which Optimum preset is the right one and
whether INT8 is faster than FP32 at all. Both `quantize_onnx.py` and
`arch_bench.py` need that answer, so it is read from the host rather than
inferred from the architecture name: x86-64 is not one instruction set, and a
GitHub Actions runner may be an Intel part with AVX-512 VNNI or an AMD part
with neither.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

# x86-64 features that give INT8 a fast path, most to least capable.
X86_FLAGS = ("amx_int8", "avx512_vnni", "avx_vnni", "avx512f", "avx2")
# The AArch64 equivalents: SDOT and the 8-bit matrix multiply extension.
ARM_FEATURES = ("FEAT_DotProd", "FEAT_I8MM", "FEAT_BF16")


def _linux_cpuinfo() -> tuple[str | None, set[str]]:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None, set()
    model: str | None = None
    flags: set[str] = set()
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "model name" and model is None:
            model = value
        elif key in {"flags", "Features"}:
            flags |= set(value.split())
    return model, flags


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_flags() -> set[str]:
    """Feature flags reported by the OS. Empty when the OS exposes none."""
    if platform.system() == "Linux":
        return _linux_cpuinfo()[1]
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        return {f for f in X86_FLAGS if _sysctl(f"hw.optional.{f}") == "1"}
    return set()


def cpu_details() -> dict:
    """Model name and the INT8-relevant features of this CPU, for a report."""
    out: dict = {
        "machine": platform.machine(),
        "system": platform.system(),
        "logical_cpus": os.cpu_count(),
        "model": None,
        "int8_features": {},
    }
    if platform.system() == "Linux":
        model, flags = _linux_cpuinfo()
        out["model"] = model
        out["int8_features"] = {f: (f in flags) for f in X86_FLAGS}
    elif platform.system() == "Darwin":
        out["model"] = _sysctl("machdep.cpu.brand_string")
        if platform.machine() == "arm64":
            out["int8_features"] = {
                f: _sysctl(f"hw.optional.arm.{f}") == "1" for f in ARM_FEATURES
            }
        else:
            flags = cpu_flags()
            out["int8_features"] = {f: (f in flags) for f in X86_FLAGS}
    return out


def isa_tag() -> str:
    """A short name for the INT8-relevant capability of this CPU.

    Used to name reports, so two runs on differently-provisioned CI machines do
    not land in the same file and get mistaken for one measurement.
    """
    if platform.machine().lower() in {"arm64", "aarch64"}:
        return "arm64-i8mm" if cpu_details()["int8_features"].get("FEAT_I8MM") else "arm64"
    flags = cpu_flags()
    for name in ("amx_int8", "avx512_vnni", "avx_vnni"):
        if name in flags:
            return f"x86-{name}"
    if "avx512f" in flags:
        return "x86-avx512"
    if "avx2" in flags:
        return "x86-avx2"
    return f"{platform.machine()}-unknown"
