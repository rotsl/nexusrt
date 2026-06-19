"""nexusrt-bench — CLI entry point. Run benchmark suites."""
from __future__ import annotations

import argparse
import json
import sys
import time

import nexusrt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nexusrt-bench")
    p.add_argument("--stage", choices=["detect", "memory", "scheduler",
                                       "pipeline", "all"],
                   default="all")
    p.add_argument("--profile", default="auto")
    p.add_argument("--size-mb", type=int, default=64,
                   help="Allocation size for memory benchmark")
    p.add_argument("--n-iter", type=int, default=100,
                   help="Iterations for scheduler benchmark")
    p.add_argument("--json", action="store_true", help="Emit JSON output")
    args = p.parse_args(argv)

    results = {}

    if args.stage in ("detect", "all"):
        info = nexusrt.firmware.init(args.profile)
        results["detect"] = {
            "vendor": info.vendor,
            "arch":   info.arch,
            "name":   info.name,
            "hbm_gb": info.hbm_bytes >> 30,
            "sm_count": info.sm_count,
            "features": info.features,
        }
        if not args.json:
            print("=== Detect ===")
            print(json.dumps(results["detect"], indent=2))

    if args.stage in ("memory", "all"):
        if not args.json: print("\n=== Memory ===")
        t0 = time.perf_counter()
        a = nexusrt.memory.alloc(args.size_mb << 20, ilc=True)
        t1 = time.perf_counter()
        nexusrt.memory.prefetch(a.ptr, a.bytes)
        t2 = time.perf_counter()
        nexusrt.memory.free(a)
        t3 = time.perf_counter()
        results["memory"] = {
            "alloc_mb": args.size_mb,
            "alloc_us": (t1 - t0) * 1e6,
            "prefetch_us": (t2 - t1) * 1e6,
            "free_us": (t3 - t2) * 1e6,
        }
        if not args.json:
            print(json.dumps(results["memory"], indent=2))

    if args.stage in ("scheduler", "all"):
        if not args.json: print("\n=== Scheduler ===")
        t0 = time.perf_counter()
        for _ in range(args.n_iter):
            try:
                nexusrt.scheduler.wait_barrier("__noop__", 0)
            except Exception:
                pass
        t1 = time.perf_counter()
        results["scheduler"] = {
            "n_iter": args.n_iter,
            "total_us": (t1 - t0) * 1e6,
            "per_iter_us": ((t1 - t0) * 1e6) / max(args.n_iter, 1),
        }
        if not args.json:
            print(json.dumps(results["scheduler"], indent=2))

    if args.stage in ("pipeline", "all"):
        if not args.json: print("\n=== Pipeline ===")
        prompt = [1, 2, 3, 4, 5]
        t0 = time.perf_counter()
        out = nexusrt.pipeline.run_infer(prompt, max_new_tokens=16)
        t1 = time.perf_counter()
        results["pipeline"] = {
            "n_new_tokens": len(out) - len(prompt),
            "total_us": (t1 - t0) * 1e6,
            "per_token_us": ((t1 - t0) * 1e6) / max(len(out) - len(prompt), 1),
        }
        if not args.json:
            print(json.dumps(results["pipeline"], indent=2))

    try:
        nexusrt.firmware.shutdown()
    except Exception:
        pass

    if args.json:
        print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
