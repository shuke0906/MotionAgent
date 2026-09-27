"""Seed helpers for deterministic Phase 4 generation."""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator


def derive_seeds(base_seed: int | None, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if base_seed is None:
        rng = random.Random()
    else:
        rng = random.Random(base_seed)
    return [rng.randrange(1, 2**31 - 1) for _ in range(count)]


@contextmanager
def seeded_torch_rng(seed: int, device: str | None = None) -> Iterator[None]:
    import torch

    devices: list[int] = []
    if device and device.startswith("cuda") and torch.cuda.is_available():
        if ":" in device:
            devices = [int(device.split(":", 1)[1])]
        else:
            devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
