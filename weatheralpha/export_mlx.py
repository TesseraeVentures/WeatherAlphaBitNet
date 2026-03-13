"""
Export WeatherAlphaBitNet to MLX format for Apple M4 Pro inference.

MLX is Apple's array framework optimised for Apple Silicon unified memory.
BitNet b1.58 maps naturally to MLX's int8/int4 quantisation.

Conversion steps:
  1. Load PyTorch checkpoint
  2. Extract quantised weight matrices (already {-1, 0, +1})
  3. Pack to int8 (or int4 for 2-weight packing)
  4. Save MLX .npz + config JSON

Usage:
  python -m weatheralpha.export_mlx --checkpoint checkpoints/best.pt --output mlx_model/
  weatheralpha-export-mlx --checkpoint checkpoints/best.pt --output mlx_model/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from weatheralpha.bitnet import BitLinear

logger = logging.getLogger(__name__)


def quantise_for_export(weight: torch.Tensor) -> tuple[np.ndarray, float]:
    """
    Quantise weight to {-1, 0, +1} and return as int8 numpy array with scale.

    Args:
        weight: (out, in) float32 weight tensor
    Returns:
        (quantised_int8, scale_float)
    """
    scale = float(weight.abs().mean().clamp(min=1e-8))
    q = (weight / scale).round().clamp(-1, 1).to(torch.int8)
    return q.numpy(), scale


def export_state_dict_to_mlx(
    state_dict: dict,
    output_dir: Path,
    arch_config: dict,
) -> dict:
    """
    Convert PyTorch state dict to MLX-compatible numpy arrays.

    BitLinear layers: weights are quantised and scales are saved separately.
    Regular layers: saved as float32.
    """
    mlx_weights = {}
    scales = {}

    for key, tensor in state_dict.items():
        # Detect BitLinear weight by checking if it could be a BitNet weight
        # (convention: BitLinear.weight keys in our naming)
        is_bitnet_weight = (
            "weight" in key
            and "norm" not in key
            and "embed" not in key
            and "pos_embed" not in key
            and tensor.dim() == 2
        )

        if is_bitnet_weight:
            q, scale = quantise_for_export(tensor.float())
            mlx_weights[key] = q
            scales[key + ".scale"] = np.float32(scale)
            logger.debug(f"  Quantised: {key} {tuple(tensor.shape)} → int8, scale={scale:.4f}")
        else:
            mlx_weights[key] = tensor.float().numpy()
            logger.debug(f"  Float32:   {key} {tuple(tensor.shape)}")

    # Merge scales into weights dict
    mlx_weights.update(scales)

    return mlx_weights


def save_mlx_model(weights: dict, output_dir: Path, arch_config: dict, metadata: dict):
    """Save MLX model as .npz + config JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights_path = output_dir / "weights.npz"
    np.savez(weights_path, **weights)
    logger.info(f"Weights saved: {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

    # Save config
    config = {
        "arch": arch_config,
        "metadata": metadata,
        "format": "weatheralpha-mlx-v1",
        "weight_dtype": "int8 (BitNet b1.58)",
        "activation_dtype": "float16 (MLX native)",
        "note": "Use weatheralpha.mlx_infer for inference on Apple Silicon",
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Config saved: {output_dir / 'config.json'}")

    # Write MLX inference stub
    infer_stub = '''"""
WeatherAlpha MLX inference stub.
Requires: pip install mlx>=0.15
"""
import mlx.core as mx
import numpy as np
import json

def load_model(model_dir: str):
    weights = dict(np.load(f"{model_dir}/weights.npz"))
    config = json.load(open(f"{model_dir}/config.json"))
    return weights, config

def predict_temperature(weights, config, era5_context, station_id: int) -> float:
    """
    Run temperature forecast.
    era5_context: (T, 7) numpy array of normalised ERA5 features
    Returns: predicted temperature in Kelvin
    """
    # TODO: implement MLX forward pass
    # This requires porting BitLinear to use mx.matmul with int8 weights
    # See: https://ml-explore.github.io/mlx/build/html/
    raise NotImplementedError(
        "MLX inference not yet implemented. "
        "Requires porting BitLinear forward pass to MLX integer matmul."
    )
'''
    with open(output_dir / "mlx_infer.py", "w") as f:
        f.write(infer_stub)

    logger.info(f"MLX inference stub: {output_dir / 'mlx_infer.py'}")


def main():
    parser = argparse.ArgumentParser(description="Export WeatherAlphaBitNet to MLX")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--arch-config", default="configs/arch_config.json")
    parser.add_argument("--output", default="mlx_model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    with open(args.arch_config) as f:
        arch_config = json.load(f)

    metadata = {
        "epoch": ckpt.get("epoch"),
        "val_mae": ckpt.get("val_mae"),
        "params_m": ckpt.get("params_m"),
    }

    logger.info("Quantising weights for MLX export...")
    mlx_weights = export_state_dict_to_mlx(state_dict, Path(args.output), arch_config)

    save_mlx_model(mlx_weights, Path(args.output), arch_config, metadata)

    # Size comparison
    n_int8 = sum(v.size for v in mlx_weights.values() if hasattr(v, "dtype") and v.dtype == np.int8)
    n_fp32 = sum(p.numel() for p in state_dict.values())
    compression = n_fp32 * 4 / max(n_int8, 1)
    logger.info(f"Compression: {n_fp32*4/1e6:.1f} MB fp32 → ~{n_int8/1e6:.1f} MB int8 ({compression:.1f}x)")
    logger.info(f"MLX model exported to: {args.output}/")


if __name__ == "__main__":
    main()
