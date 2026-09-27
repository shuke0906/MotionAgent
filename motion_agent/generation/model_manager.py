"""GEM model loading and health checks for Phase 4."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GENMO_ROOT = REPO_ROOT / "vendor" / "GENMO"
if GENMO_ROOT.exists() and str(GENMO_ROOT) not in sys.path:
    sys.path.insert(0, str(GENMO_ROOT))


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class GEMModelManager:
    def __init__(self, checkpoint_path: str | Path, *, device: str = "cuda") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self._model = None

    @property
    def checkpoint_version(self) -> str:
        return str(self.checkpoint_path)

    def load(self):
        if self._model is not None:
            return self._model
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GEM checkpoint not found: {self.checkpoint_path}")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real GEM generation")
        from scripts.demo.demo_utils import load_model

        with _cwd(GENMO_ROOT):
            model = load_model(str(self.checkpoint_path), load_text_encoder=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model
        return self._model

    def health(self) -> dict[str, object]:
        model = self.load()
        return {
            "loaded": model is not None,
            "eval": not model.training,
            "all_weights_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
