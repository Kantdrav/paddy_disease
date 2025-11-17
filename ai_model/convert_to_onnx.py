"""
Export the trained PyTorch model checkpoint (model.pth) to ONNX.

Pipeline: PyTorch (ResNet18) -> ONNX

This script reconstructs the model architecture used in app.py:
- torchvision.models.resnet18 with ImageNet weights
- replaces the final FC to match the number of classes from the checkpoint

Usage (checkpoint present):
    python convert_to_onnx.py \
            --checkpoint model.pth \
            --onnx-path model.onnx \
            --opset 13 \
            --img-size 224

Fallback (no checkpoint, specify classes):
    python convert_to_onnx.py \
            --onnx-path model.onnx \
            --num-classes 3 \
            --class-names classA classB classC

Notes:
- Batch size is dynamic in the exported model (N x 3 x H x W).
- Saves the classes list next to the ONNX file as <onnx-path>.classes.json
- If `model.pth` is missing or malformed, the script will explain and exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple, Optional, Dict, Any


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_checkpoint(ckpt_path: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Load the training checkpoint and return (state_dict, classes).

    The checkpoint may be either:
    - a dict with keys: 'model_state' and optional 'classes'
    - a plain state dict
    - a dict with other common keys where state dict is top-level
    """
    try:
        import torch  # imported locally to keep import errors clear
    except Exception as exc:
        eprint("PyTorch is required to run this conversion.")
        eprint(exc)
        sys.exit(1)

    if not os.path.exists(ckpt_path):
        # Missing checkpoint handled by caller (fallback mode)
        return None, []

    obj = torch.load(ckpt_path, map_location="cpu")

    state_dict: Dict[str, Any]
    classes: List[str] = []

    if isinstance(obj, dict):
        # preferred format
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            state_dict = obj["model_state"]
            classes = list(obj.get("classes", []))
        else:
            # maybe it's already a state_dict
            # heuristic: look for typical layer keys like 'fc.weight'
            has_tensor_vals = any(
                isinstance(v, object) and hasattr(v, "size") for v in obj.values()
            )
            if has_tensor_vals:
                state_dict = obj  # type: ignore[assignment]
            else:
                # unknown structure
                eprint("Unsupported checkpoint structure. Expected a state_dict or a dict with 'model_state'.")
                keys = list(obj.keys())[:20]
                eprint(f"Top-level keys: {keys}")
                sys.exit(1)
    else:
        eprint("Unsupported checkpoint type. Expected dict/state_dict.")
        sys.exit(1)

    # If classes missing, attempt to infer from fc.weight shape
    if not classes:
        num_classes: Optional[int] = None
        fc_w = state_dict.get("fc.weight")
        try:
            num_classes = int(fc_w.shape[0]) if fc_w is not None else None
        except Exception:
            num_classes = None
        if num_classes is not None and num_classes > 0:
            classes = [f"class_{i}" for i in range(num_classes)]

    return state_dict, classes


def build_model(num_classes: int):
    """Construct resnet18 and swap the final FC to num_classes.

    Mirrors the definition in app.py for compatibility with the checkpoint.
    """
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except Exception as exc:
        eprint("torch and torchvision are required to reconstruct the model.")
        eprint(exc)
        sys.exit(1)

    # Use default weights for better export graph stability; not strictly required
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, max(1, int(num_classes)))
    model.eval()
    return model


def export_onnx(
    state_dict: Dict[str, Any],
    classes: List[str],
    onnx_path: str,
    opset: int = 13,
    img_size: int = 224,
) -> None:
    try:
        import torch
    except Exception as exc:
        eprint("PyTorch is required to export ONNX.")
        eprint(exc)
        sys.exit(1)

    if img_size <= 0:
        eprint("--img-size must be a positive integer")
        sys.exit(1)

    num_classes = len(classes) if classes else None
    # If classes are unknown, try infer from state_dict
    if not num_classes:
        fc_w = state_dict.get("fc.weight")
        try:
            num_classes = int(fc_w.shape[0]) if fc_w is not None else None
        except Exception:
            num_classes = None

    if not num_classes:
        eprint("Unable to determine number of classes from checkpoint.\n"
               "- Provide a checkpoint that includes 'classes' list, or\n"
               "- Ensure state_dict has 'fc.weight' with correct out_features.")
        sys.exit(1)

    model = build_model(num_classes)

    # Load weights (allow missing/buffer mismatches gracefully)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        eprint(f"Warning: missing keys when loading state_dict: {missing}")
    if unexpected:
        eprint(f"Warning: unexpected keys in state_dict: {unexpected}")

    model.eval()

    # Dummy input for tracing
    dummy = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)) or ".", exist_ok=True)

    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # If future exporter dependencies (onnxscript, onnx) are missing, provide guidance.
    # The above call may raise ModuleNotFoundError earlier; add proactive check here for clarity.
    try:  # lightweight advisory
        import importlib
        for pkg in ("onnx", "onnxscript"):
            if importlib.util.find_spec(pkg) is None:
                print(f"[advice] Optional package '{pkg}' not found. Install with: pip install {pkg}")
    except Exception:
        pass

    # Save classes next to the ONNX file
    classes_out = os.path.splitext(onnx_path)[0] + ".classes.json"
    with open(classes_out, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)

    print("Export complete:")
    print(f"  ONNX:    {onnx_path}")
    print(f"  Classes: {classes_out}  (num_classes={len(classes)})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PyTorch checkpoint to ONNX")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="model.pth",
        help="Path to the PyTorch checkpoint (default: model.pth). If missing, must supply --num-classes.",
    )
    p.add_argument(
        "--onnx-path",
        type=str,
        default="model.onnx",
        help="Destination path for ONNX file (default: model.onnx)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size (HxW) (default: 224)",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes if checkpoint absent (fallback).",
    )
    p.add_argument(
        "--class-names",
        nargs="*",
        help="Optional explicit class names matching --num-classes when checkpoint is absent.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    state_dict, classes = load_checkpoint(args.checkpoint)
    if state_dict is None:
        # Fallback path: user must provide num_classes
        if args.num_classes is None:
            eprint("Checkpoint missing and --num-classes not provided. Aborting.")
            sys.exit(1)
        if args.class_names and len(args.class_names) != args.num_classes:
            eprint("Length of --class-names must match --num-classes")
            sys.exit(1)
        if not args.class_names:
            classes = [f"class_{i}" for i in range(args.num_classes)]
        else:
            classes = args.class_names
        # Build a random model to obtain a state_dict structure
        model_tmp = build_model(len(classes))
        try:
            import torch
        except Exception as exc:
            eprint("PyTorch required for fallback model build.")
            eprint(exc)
            sys.exit(1)
        state_dict = model_tmp.state_dict()
        print("Checkpoint missing. Proceeding with randomly initialized model (no trained weights).")
    export_onnx(state_dict, classes, args.onnx_path, args.opset, args.img_size)


if __name__ == "__main__":
    main()
