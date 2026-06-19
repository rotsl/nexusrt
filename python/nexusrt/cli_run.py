"""nexusrt-run — CLI entry point. Run an end-to-end pipeline."""
from __future__ import annotations

import argparse
import json
import sys

import nexusrt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nexusrt-run")
    p.add_argument("--pipeline", default="config/pipeline.yaml",
                   help="Path to pipeline YAML")
    p.add_argument("--hardware", default="config/hardware.yaml",
                   help="Path to hardware YAML")
    p.add_argument("--runtime", default="config/runtime.yaml",
                   help="Path to runtime YAML")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--profile", default="auto",
                   help="Hardware profile (a100 | h100 | m1pro | auto)")
    p.add_argument("--prompt", default="Hello, NexusRT.",
                   help="Prompt for inference")
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args(argv)

    info = nexusrt.firmware.init(args.profile)
    print(f"[nexusrt] device: {info.vendor}/{info.arch} — {info.name}")
    print(f"[nexusrt] HBM: {info.hbm_bytes >> 30} GB, SMs: {info.sm_count}")
    print(f"[nexusrt] features: {info.features}")

    # Encode prompt as UTF-8 bytes → token IDs (stub tokenizer).
    tokens = list(args.prompt.encode("utf-8"))
    out = nexusrt.pipeline.run_infer(tokens, args.max_tokens)
    text = nexusrt.pipeline.run_postprocess(out)
    print(f"[nexusrt] generated {len(out) - len(tokens)} new tokens")
    print(f"[nexusrt] final: {text!r}")

    nexusrt.firmware.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
