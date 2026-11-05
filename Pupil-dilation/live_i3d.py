import os
import sys
from collections import deque
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

BACKEND_DIR = Path(__file__).resolve().parent
I3D_DIR = Path(
    os.environ.get("I3D_LIBRARY_DIR", BACKEND_DIR / "external" / "video_features-master")
).expanduser().resolve()

if not (I3D_DIR / "models/i3d/checkpoints/i3d_rgb.pt").is_file():
    raise FileNotFoundError(f"I3D RGB checkpoint not found in {I3D_DIR}")

if str(I3D_DIR) not in sys.path:
    sys.path.insert(0, str(I3D_DIR))

from models.i3d.extract_i3d import ExtractI3D


class LiveI3D:
    """Extract RGB I3D features from 65 captured frames (64 model frames)."""

    frames_required = 65

    def __init__(self):
        self.frames = deque(maxlen=self.frames_required)
        self.extractor = None
        self.lock = __import__("threading").Lock()
        self.last_sample_at = None
        self.sample_interval_seconds = 0.5
        self.inference_executor = ThreadPoolExecutor(max_workers=1)
        self.inference_future: Future | None = None
        self.inference_generation = 0
        self.generation = 0

        cfg = OmegaConf.load(str(I3D_DIR / "configs/i3d.yml"))
        cfg.device = "cpu"
        cfg.streams = "rgb"
        cfg.video_paths = None
        cfg.flow_type = "raft"
        cfg.on_extraction = "print"
        cfg.output_path = str(I3D_DIR / "output")
        cfg.tmp_path = str(I3D_DIR / "tmp")

        # The upstream extractor uses relative checkpoint paths.
        old_cwd = os.getcwd()
        try:
            os.chdir(I3D_DIR)
            self.extractor = ExtractI3D(cfg)
        finally:
            os.chdir(old_cwd)

    @property
    def frames_collected(self):
        with self.lock:
            return len(self.frames)

    @property
    def inference_pending(self):
        with self.lock:
            return self.inference_future is not None

    def reset(self):
        with self.lock:
            self.frames.clear()
            self.last_sample_at = None
            self.generation += 1

    def _extract_stack(self, stack):
        # Match the upstream extractor's 64-frame RGB stack.
        with torch.inference_mode():
            output = self.extractor.run_on_a_stack(
                stack, stack_counter=0, padder=None
            )
        embedding = output["rgb"].detach().cpu().numpy().reshape(-1)
        if embedding.size != 1024 or not np.isfinite(embedding).all():
            raise ValueError(
                f"Unexpected I3D feature shape or values: {embedding.shape}"
            )
        return embedding.astype(np.float32).tolist()

    def process_frame(self, frame_bgr):
        """Collect samples at 2 fps and run I3D asynchronously after each stack."""
        with self.lock:
            embedding = None
            if self.inference_future is not None and self.inference_future.done():
                completed_generation = self.inference_generation
                embedding = self.inference_future.result()
                self.inference_future = None
                if completed_generation != self.generation:
                    embedding = None

            now = __import__("time").monotonic()
            if (
                self.last_sample_at is None
                or now - self.last_sample_at >= self.sample_interval_seconds
            ):
                self.last_sample_at = now
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                resized = self.extractor.resize_transforms(rgb)
                self.frames.append(resized.unsqueeze(0))

            if (
                len(self.frames) >= self.frames_required
                and self.inference_future is None
            ):
                stack = list(self.frames)
                self.frames.clear()
                self.inference_generation = self.generation
                self.inference_future = self.inference_executor.submit(
                    self._extract_stack, stack
                )

            return embedding
