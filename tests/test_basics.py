"""Fast checks (no GPU/model needed):  python tests/test_basics.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.engine import GenerationRequest, VideoEngine, snap_dim, snap_frames  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def test_snap():
    assert snap_frames(121) == 121 and snap_frames(120) == 121 and snap_frames(81) == 81 and snap_frames(2) == 5
    assert snap_dim(704) == 704 and snap_dim(720) == 704 and snap_dim(1080) == 1088 and snap_dim(1280) == 1280


def test_profiles_load():
    prod = load_config("config/prod_rtx5080.toml")
    dev = load_config("config/dev_rtx4050.toml")
    assert prod.model.backend == "wan" and prod.memory.offload == "model" and prod.device.expect_gpu_name == "5080"
    assert (prod.generation.width, prod.generation.height, prod.generation.num_frames) == (1280, 704, 121)
    assert dev.model.backend == "tiny"
    assert prod.model.revision == dev.model.revision  # same pinned weights on both machines


def test_duration_to_frames():
    eng = VideoEngine(load_config("config/prod_rtx5080.toml"))
    assert eng._resolve(GenerationRequest(prompt="x", duration=5))["num_frames"] == 121
    assert eng._resolve(GenerationRequest(prompt="x", duration=10))["num_frames"] == 121  # clamped
    assert eng._resolve(GenerationRequest(prompt="x", duration=10, allow_long=True))["num_frames"] == 241
    assert eng._resolve(GenerationRequest(prompt="x", duration=3))["num_frames"] == 73


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parents[1])
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
