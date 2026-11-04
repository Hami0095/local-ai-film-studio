"""Hardware / environment audit.  Writes reports/hardware_audit_<host>.md + .json.

Run on BOTH machines: Machine A to document the dev box, Machine B to confirm
the RTX 5080 deployment meets the production profile's requirements.
"""
from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil

from ..utils.config import PROJECT_ROOT, StudioConfig
from ..utils.logging_setup import get_logger

log = get_logger("audit")

# Requirements for the production profile (Wan2.2 TI2V-5B, 1280x704x121, model CPU offload)
PROD_REQUIREMENTS = {
    "min_vram_gb": 15.0,           # RTX 5080 reports ~15.9 GiB usable
    "min_ram_gb": 32.0,            # offloaded components live in system RAM (~25 GB bf16/fp32)
    "min_free_disk_gb": 45.0,      # 34.2 GB weights + outputs + headroom
    "min_compute_capability": (8, 0),
    "min_driver_major": 570,       # Blackwell + CUDA 12.8 wheels
}
# The `episode` pipeline adds FLUX.1-schnell (~34 GB), ACE-Step (8.3 GB) + its venv (~6 GB), Kokoro (0.4 GB)
EPISODE_MIN_FREE_DISK_GB = 110.0

PACKAGES = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "huggingface_hub",
            "safetensors", "sentencepiece", "ftfy", "imageio", "imageio-ffmpeg", "numpy", "pillow",
            "xformers", "flash-attn", "sageattention"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def _cpu_name() -> str:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if out:
            return out.splitlines()[0].strip()
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _nvidia_smi() -> list[dict[str, str]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap,pcie.link.gen.max",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append(dict(index=parts[0], name=parts[1], vram_total_mib=parts[2], vram_used_mib=parts[3],
                             driver=parts[4], compute_cap=parts[5]))
    header = _run([exe]) or ""
    cuda_line = next((l for l in header.splitlines() if "CUDA Version" in l), "")
    if cuda_line and gpus:
        gpus[0]["driver_max_cuda"] = cuda_line.split("CUDA Version:")[-1].strip(" |")
    return gpus


def _other_gpus() -> list[str]:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"])
        return [l.strip() for l in (out or "").splitlines() if l.strip()]
    out = _run(["lspci"]) or ""
    return [l for l in out.splitlines() if "VGA" in l or "3D controller" in l]


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": str(exc)}
    info: dict[str, Any] = {"installed": True, "version": torch.__version__, "cuda_build": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                            "cuda_available": torch.cuda.is_available(), "arch_list": torch.cuda.get_arch_list()
                            if torch.cuda.is_available() else [], "devices": []}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            info["devices"].append(dict(index=i, name=p.name, capability=f"{p.major}.{p.minor}",
                                        total_gib=round(total / 2**30, 2), free_gib=round(free / 2**30, 2),
                                        sm_count=p.multi_processor_count,
                                        bf16=torch.cuda.is_bf16_supported()))
        # actually run a kernel on each GPU - proves the build supports this architecture
        try:
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            info["kernel_test"] = f"ok (matmul checksum {float((x @ x).float().abs().mean()):.3f})"
        except Exception as exc:
            info["kernel_test"] = f"FAILED: {exc}"
    return info


def _pkg_versions() -> dict[str, str | None]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def _other_envs() -> list[str]:
    envs = []
    conda = shutil.which("conda")
    if conda:
        out = _run([conda, "env", "list"]) or ""
        envs += [f"conda: {l.split()[0]}" for l in out.splitlines() if l and not l.startswith("#") and l.split()]
    for name in ["python", "python3", "py"]:
        p = shutil.which(name)
        if p:
            envs.append(f"{name}: {p}")
    return envs


def collect(cfg: StudioConfig) -> dict[str, Any]:
    from ..utils.ffmpeg_tools import ffmpeg_version

    model_dir = cfg.resolve(cfg.model.local_dir)
    anchor = model_dir if model_dir.exists() else PROJECT_ROOT
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(anchor)
    try:
        ffv = ffmpeg_version()
    except Exception as exc:
        ffv = f"not available: {exc}"
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": _cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "disk_path": str(anchor),
        "disk_free_gb": round(du.free / 1e9, 1),
        "disk_total_gb": round(du.total / 1e9, 1),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "nvidia_smi": _nvidia_smi(),
        "all_display_adapters": _other_gpus(),
        "torch": _torch_info(),
        "packages": _pkg_versions(),
        "ffmpeg": ffv,
        "other_python_envs": _other_envs(),
        "profile": cfg.profile,
    }


def evaluate_for_production(a: dict[str, Any]) -> list[tuple[str, bool, str]]:
    r = PROD_REQUIREMENTS
    rows: list[tuple[str, bool, str]] = []
    devs = a["torch"].get("devices", [])
    d0 = devs[0] if devs else None
    rows.append(("CUDA available in torch", bool(a["torch"].get("cuda_available")), str(a["torch"].get("cuda_build"))))
    rows.append(("GPU kernel executes", str(a["torch"].get("kernel_test", "")).startswith("ok"), a["torch"].get("kernel_test", "n/a")))
    rows.append((f"VRAM >= {r['min_vram_gb']} GiB", bool(d0 and d0["total_gib"] >= r["min_vram_gb"]),
                 f"{d0['total_gib']} GiB" if d0 else "no GPU"))
    rows.append((f"System RAM >= {r['min_ram_gb']} GB", a["ram_total_gb"] >= r["min_ram_gb"], f"{a['ram_total_gb']} GB"))
    rows.append((f"Free disk >= {r['min_free_disk_gb']} GB (or model already present)", a["disk_free_gb"] >= r["min_free_disk_gb"],
                 f"{a['disk_free_gb']} GB at {a['disk_path']}"))
    drv = a["nvidia_smi"][0]["driver"] if a["nvidia_smi"] else "0"
    rows.append((f"Free disk >= {EPISODE_MIN_FREE_DISK_GB:.0f} GB for the episode pipeline (optional)",
                 a["disk_free_gb"] >= EPISODE_MIN_FREE_DISK_GB, f"{a['disk_free_gb']} GB"))
    rows.append((f"NVIDIA driver >= {r['min_driver_major']}", int(drv.split(".")[0]) >= r["min_driver_major"], drv))
    rows.append(("FFmpeg available", not a["ffmpeg"].startswith("not available"), a["ffmpeg"][:60]))
    return rows


def write_report(a: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    host = a["hostname"].replace(" ", "_")
    (out_dir / f"hardware_audit_{host}.json").write_text(json.dumps(a, indent=2), encoding="utf-8")
    L = [f"# Hardware audit - {a['hostname']}", "", f"Generated {a['timestamp']} by `python main.py audit` (profile `{a['profile']}`).", "",
         "| Item | Value |", "|---|---|",
         f"| OS | {a['os']} |", f"| CPU | {a['cpu']} ({a['cpu_physical_cores']} cores / {a['cpu_logical_cores']} threads) |",
         f"| System RAM | {a['ram_total_gb']} GB total, {a['ram_available_gb']} GB available |",
         f"| Disk ({a['disk_path']}) | {a['disk_free_gb']} GB free of {a['disk_total_gb']} GB |",
         f"| Python | {a['python']} ({a['python_executable']}) |", f"| FFmpeg | {a['ffmpeg']} |",
         f"| Display adapters | {'; '.join(a['all_display_adapters'])} |", ""]
    L += ["## NVIDIA GPUs (nvidia-smi)", "", "| # | Name | VRAM MiB | Driver | Compute cap | Driver max CUDA |", "|---|---|---|---|---|---|"]
    for g in a["nvidia_smi"]:
        L.append(f"| {g['index']} | {g['name']} | {g['vram_total_mib']} | {g['driver']} | {g['compute_cap']} | {g.get('driver_max_cuda', '')} |")
    t = a["torch"]
    L += ["", "## PyTorch (project venv)", ""]
    if t.get("installed"):
        L += [f"- torch {t['version']}, CUDA build {t['cuda_build']}, cuDNN {t['cudnn']}, cuda_available={t['cuda_available']}",
              f"- compiled arch list: {', '.join(t['arch_list'])}", f"- kernel test: {t.get('kernel_test', 'n/a')}"]
        for d in t["devices"]:
            L.append(f"- cuda:{d['index']} {d['name']} sm_{d['capability'].replace('.', '')} {d['total_gib']} GiB "
                     f"({d['free_gib']} free), {d['sm_count']} SMs, bf16={d['bf16']}")
    else:
        L.append(f"- torch not installed: {t.get('error')}")
    L += ["", "## Python packages", "", "| Package | Version |", "|---|---|"]
    L += [f"| {k} | {v or '-'} |" for k, v in a["packages"].items()]
    L += ["", "## Other Python environments on this machine", ""] + [f"- {e}" for e in a["other_python_envs"]]
    L += ["", "## Production readiness (Wan2.2 TI2V-5B, 1280x704, 121 frames)", "", "| Check | Pass | Detail |", "|---|---|---|"]
    for name, ok, detail in evaluate_for_production(a):
        L.append(f"| {name} | {'yes' if ok else 'NO'} | {detail} |")
    path = out_dir / f"hardware_audit_{host}.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
