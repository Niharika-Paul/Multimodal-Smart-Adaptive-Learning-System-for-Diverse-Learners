from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "CMOSE_W0P_checkpoints_export_20261027"
DEVICE = torch.device("cpu")

LABELS = [
    "Highly Disengaged",
    "Disengaged",
    "Engaged",
    "Highly Engaged",
]


class EngagementModel:
    """Loads a saved W0 or W0-P classifier for 1024-D embeddings."""

    def __init__(self, checkpoint_path: Path):
        self.path = checkpoint_path
        self.checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False
        )

        if self.checkpoint.get("method") == "W0-P":
            base_path = (
                checkpoint_path.parent
                / self.checkpoint["base_w0_checkpoint"]
            ).resolve()
            base = torch.load(
                base_path, map_location=DEVICE, weights_only=False
            )
            state = dict(base["model_state"])
            state.update({
                f"classifier.{key}": value
                for key, value in self.checkpoint["classifier_state"].items()
            })
            self.mean = torch.as_tensor(base["scaler_mean"], dtype=torch.float32)
            self.scale = torch.as_tensor(base["scaler_scale"], dtype=torch.float32)
        else:
            state = self.checkpoint["model_state"]
            self.mean = torch.as_tensor(
                self.checkpoint["scaler_mean"], dtype=torch.float32
            )
            self.scale = torch.as_tensor(
                self.checkpoint["scaler_scale"], dtype=torch.float32
            )

        self.layers = []
        for key, weight in state.items():
            if key.endswith(".weight") and weight.ndim == 2:
                prefix = key[:-len(".weight")]
                bias = state.get(f"{prefix}.bias")
                if bias is not None:
                    self.layers.append((weight.float(), bias.float()))

        # Order the linear layers by their dimensions: 1024 -> 256 -> 4.
        self.layers.sort(key=lambda layer: -layer[0].shape[1])
        if not self.layers or self.layers[0][0].shape[1] != 1024:
            raise ValueError("Unexpected model architecture or state dictionary")

        self.mean = self.mean.reshape(-1)
        self.scale = self.scale.reshape(-1)
        self.scale = torch.where(self.scale == 0, 1.0, self.scale)

    @torch.inference_mode()
    def predict(self, embedding):
        x = torch.as_tensor(embedding, dtype=torch.float32).reshape(-1)
        if x.numel() != 1024:
            raise ValueError("Expected exactly 1024 embedding values")
        if not torch.isfinite(x).all():
            raise ValueError("Embedding contains non-finite values")

        x = (x - self.mean) / self.scale

        for index, (weight, bias) in enumerate(self.layers):
            x = F.linear(x, weight, bias)
            if index < len(self.layers) - 1:
                x = F.relu(x)

        probabilities = torch.softmax(x, dim=-1)
        index = int(probabilities.argmax().item())

        return {
            "label": LABELS[index],
            "class_index": index,
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                label: float(probabilities[i].item())
                for i, label in enumerate(LABELS)
            },
            "model": self.checkpoint.get("method", "W0"),
        }


def load_w0p(filename: str) -> EngagementModel:
    path = (CHECKPOINT_DIR / "w0p_checkpoints" / filename).resolve()
    if not path.is_relative_to(CHECKPOINT_DIR.resolve()):
        raise ValueError("Invalid checkpoint path")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    return EngagementModel(path)
