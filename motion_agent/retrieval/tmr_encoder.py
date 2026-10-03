"""TMR-compatible encoder boundary.

The production Phase 5 path expects official TMR latents. This module keeps the
runtime boundary explicit and provides a deterministic tiny encoder for local
fixtures only; it is not used to claim the real HumanML3D/TMR stage gate.
"""

from __future__ import annotations

import math
import re


class OfficialTMREncoder:
    """Load official weights, DistilBERT tokens, and training-set normalization."""

    def __init__(self, repository: str, run_dir: str, device: str = "cpu"):
        self.repository, self.run_dir, self.device = repository, run_dir, device
        self.model = self.text_model = self.normalizer = self.collate = None

    def cache_identity(self):
        from pathlib import Path
        from motion_agent.verification.artifacts import file_digest
        paths = [Path(self.run_dir) / "config.json", Path(self.run_dir) / "last_weights/motion_encoder.pt",
                 Path(self.run_dir) / "last_weights/text_encoder.pt",
                 Path(self.repository) / "stats/humanml3d/guoh3dfeats/mean.pt",
                 Path(self.repository) / "stats/humanml3d/guoh3dfeats/std.pt",
                 Path(self.repository) / "src/guofeats/motion_representation.py",
                 Path(self.repository) / "src/guofeats/skeleton_example_h3d.npy"]
        return {"model": self.run_dir, "device": self.device,
                "assets": {str(p): file_digest(p) if p.is_file() else "missing" for p in paths},
                "code": "official_tmr_encoder_v1"}

    def load(self):
        if self.model is not None and self.text_model is not None and self.normalizer is not None:
            return
        import importlib
        from pathlib import Path
        from motion_agent.verification.backends import BackendUnavailable
        from motion_agent.verification.representations import official_repository
        run_dir = Path(self.run_dir).resolve()
        required = ["config.json", "last_weights/motion_encoder.pt", "last_weights/text_encoder.pt"]
        missing = [str(run_dir / name) for name in required if not (run_dir / name).is_file()]
        if missing:
            raise BackendUnavailable("official TMR assets missing: " + ", ".join(missing))
        with official_repository(self.repository, "src"):
            from hydra.utils import instantiate
            cfg = importlib.import_module("src.config").read_config(str(run_dir))
            cfg.run_dir = str(run_dir)
            model = importlib.import_module("src.load").load_model_from_cfg(cfg, device=self.device, eval_mode=True)
            normalizer_cfg = cfg.data.motion_loader.normalizer
            if "base_dir" in normalizer_cfg:
                normalizer_cfg.base_dir = str(Path(self.repository).resolve() / normalizer_cfg.base_dir)
            normalizer = instantiate(normalizer_cfg)
            text_model = instantiate(cfg.data.text_to_token_emb, device=self.device, preload=False)
            self.collate = importlib.import_module("src.data.collate").collate_x_dict
            self.model, self.normalizer, self.text_model = model, normalizer, text_model

    def similarity(self, text, features) -> float:
        import torch
        self.load()
        motion = self.normalizer(features).to(self.device)
        with torch.inference_mode():
            motion_latent = self.model.encode(self.collate([{"x": motion, "length": len(motion)}]), sample_mean=True)
            text_latent = self.model.encode(self.collate(self.text_model([text])), sample_mean=True)
            cosine = torch.nn.functional.cosine_similarity(text_latent, motion_latent, dim=-1)
            score = (cosine + 1) / 2
        if score.numel() != 1 or not torch.isfinite(score).all():
            raise ValueError("TMR returned an invalid similarity")
        return float(score.item())


class DeterministicTextEncoder:
    """Small deterministic text encoder for unit fixtures."""

    terms = [
        "walk",
        "limp",
        "sit",
        "jump",
        "wave",
        "turn",
        "forward",
        "backward",
        "right",
        "left",
        "hand",
        "arm",
        "leg",
        "slow",
        "fast",
        "asymmetric",
    ]

    aliases = {
        "walking": "walk",
        "walks": "walk",
        "limping": "limp",
        "limps": "limp",
        "seated": "sit",
        "sitting": "sit",
        "sits": "sit",
        "jumping": "jump",
        "jumps": "jump",
        "waving": "wave",
        "waves": "wave",
        "turning": "turn",
        "turns": "turn",
        "uneven": "asymmetric",
        "injured": "limp",
        "gait": "walk",
    }

    def encode_text(self, text: str) -> list[float]:
        tokens = [self.aliases.get(token, token) for token in re.findall(r"[a-zA-Z_]+", text.lower())]
        vector = [0.0 for _ in self.terms]
        for token in tokens:
            if token in self.terms:
                vector[self.terms.index(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def encode_motion_caption(self, captions: list[str]) -> list[float]:
        return self.encode_text(" ".join(captions))
