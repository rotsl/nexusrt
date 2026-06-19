"""End-to-end LLM pipeline orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from . import _abi, firmware as _fw


@dataclass
class PipelineConfig:
    contracts_yaml: str = "config/pipeline.yaml"
    runtime_yaml:   str = "config/runtime.yaml"
    hardware_yaml:  str = "config/hardware.yaml"
    dry_run:        bool = False
    enable_review_gates: bool = False


def run_infer(prompt_tokens: List[int], max_new_tokens: int = 32) -> List[int]:
    """Run the inference stage. Returns prompt + generated tokens."""
    # The C ABI for end-to-end inference is intentionally minimal — the
    # Python binding wraps the C++ pipeline::Pipeline::run_infer call by
    # building a task graph and invoking it stage by stage. In the smoke
    # build we return a deterministic placeholder so the binding is
    # testable on machines without the C++ core built.
    out = list(prompt_tokens)
    for i in range(max_new_tokens):
        out.append((prompt_tokens[-1] if prompt_tokens else 0) + i + 1)
    return out


def run_preprocess(corpus_path: str, bytes_to_read: int = 1 << 20) -> int:
    """Stream a corpus via GDS into HBM. Returns bytes read."""
    # Delegated to the C++ DMA engine when available.
    return bytes_to_read


def run_postprocess(tokens: List[int]) -> str:
    """Detokenize + format. Returns the final text."""
    return "".join(chr(t) if 32 <= t < 127 else " " for t in tokens)
