"""
Convert an ONNX model to a TFLite model for lightweight deployment.

Expected inputs:
- model.onnx (exported via convert_to_onnx.py)
- model.classes.json (labels)

Pipeline:
  ONNX -> TensorFlow (SavedModel) via onnx2tf -> TFLite (.tflite) via TF Lite Converter

NOTE: This script requires TensorFlow and onnx2tf at conversion time only.
      Do NOT add these heavy packages to production requirements.
      Run this locally, commit the resulting .tflite and classes.json, and
      deploy with tflite-runtime in requirements.

Usage:
  python convert_to_tflite.py \
    --onnx ai_model/model.onnx \
    --saved-model ai_model/export_tf \
    --tflite ai_model/model.tflite \
    --optimize float16

Optimization options: none|default|float16|int8
"""

from __future__ import annotations

import argparse
import os
import sys


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def convert_onnx_to_tf(onnx_path: str, saved_model_dir: str) -> None:
    try:
        from onnx2tf import convert
    except Exception as exc:
        eprint("onnx2tf is required for ONNX->TF conversion. Install with: pip install onnx2tf")
        eprint(exc)
        sys.exit(1)

    os.makedirs(saved_model_dir, exist_ok=True)
    # onnx2tf writes a SavedModel into --output_signaturedefs parameter path.
    convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        copy_onnx_input_output_names_to_tflite=True,
        output_integer_quantized_tflite=False,
    )


def convert_tf_to_tflite(saved_model_dir: str, tflite_path: str, optimize: str) -> None:
    try:
        import tensorflow as tf
    except Exception as exc:
        eprint("TensorFlow is required for TF->TFLite conversion. Install with: pip install 'tensorflow<2.16'")
        eprint(exc)
        sys.exit(1)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if optimize in {"default", "float16", "int8"}:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if optimize == "float16":
        converter.target_spec.supported_types = [tf.float16]
    # For int8, post-training quantization would need a calibration dataset; omitted here.

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(tflite_path)) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote TFLite model -> {tflite_path}")


def parse_args():
    p = argparse.ArgumentParser(description="ONNX -> TFLite converter")
    p.add_argument("--onnx", default="ai_model/model.onnx", help="Path to ONNX model")
    p.add_argument("--saved-model", default="ai_model/export_tf", help="Output dir for SavedModel")
    p.add_argument("--tflite", default="ai_model/model.tflite", help="Output path for TFLite model")
    p.add_argument(
        "--optimize",
        choices=["none", "default", "float16", "int8"],
        default="default",
        help="TFLite optimization level",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.onnx):
        eprint(f"ONNX file not found: {args.onnx}")
        sys.exit(1)
    convert_onnx_to_tf(args.onnx, args.saved_model)
    convert_tf_to_tflite(args.saved_model, args.tflite, args.optimize)


if __name__ == "__main__":
    main()
