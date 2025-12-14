#!/usr/bin/env python3
"""
Combined GPU Test Suite for RingRift AI

Runs all GPU-related tests:
1. GPU Minimax AI tests
2. GPU Data Generation tests

Usage:
    PYTHONPATH=. python scripts/test_gpu_all.py --quick       # Quick validation
    PYTHONPATH=. python scripts/test_gpu_all.py --full        # Full test suite
    PYTHONPATH=. python scripts/test_gpu_all.py --benchmark   # Performance only

For cloud deployment:
    # On cloud GPU instance (Lambda H100, etc.)
    cd /path/to/RingRift/ai-service
    pip install -r requirements.txt
    PYTHONPATH=. python scripts/test_gpu_all.py --full 2>&1 | tee gpu_test_results.log
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

# Ensure ai-service is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.test_gpu_minimax import GPUMinimaxTester
from scripts.test_gpu_data_generation import GPUDataGenTester


def get_system_info() -> dict:
    """Gather system information for test report."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["gpu_count"] = torch.cuda.device_count()

    return info


def run_all_tests(mode: str = "quick", verbose: bool = True) -> dict:
    """Run all GPU tests and return results."""
    print("=" * 70)
    print("RINGRIFT GPU TEST SUITE")
    print("=" * 70)
    print()

    # System info
    sys_info = get_system_info()
    print("System Information:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    print()

    results = {
        "system_info": sys_info,
        "mode": mode,
        "start_time": datetime.now().isoformat(),
        "tests": {},
    }

    total_start = time.time()

    # Run GPU Minimax tests
    print("\n" + "=" * 70)
    print("PART 1: GPU MINIMAX AI TESTS")
    print("=" * 70)

    minimax_tester = GPUMinimaxTester(verbose=verbose)

    if mode == "benchmark":
        minimax_passed = minimax_tester.run_benchmark()
    elif mode == "full":
        minimax_passed = minimax_tester.run_full_tests()
    else:
        minimax_passed = minimax_tester.run_quick_tests()

    results["tests"]["gpu_minimax"] = {
        "passed": minimax_passed,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "duration_sec": r.duration_sec,
                "metrics": r.metrics,
            }
            for r in minimax_tester.results
        ],
    }

    # Run GPU Data Generation tests
    print("\n" + "=" * 70)
    print("PART 2: GPU DATA GENERATION TESTS")
    print("=" * 70)

    datagen_tester = GPUDataGenTester(verbose=verbose)

    if mode == "benchmark":
        datagen_passed = datagen_tester.run_benchmark()
    elif mode == "full":
        datagen_passed = datagen_tester.run_full_tests()
    else:
        datagen_passed = datagen_tester.run_quick_tests()

    results["tests"]["gpu_data_generation"] = {
        "passed": datagen_passed,
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "message": r.message,
                "duration_sec": r.duration_sec,
                "metrics": r.metrics,
            }
            for r in datagen_tester.results
        ],
    }

    total_duration = time.time() - total_start

    results["end_time"] = datetime.now().isoformat()
    results["total_duration_sec"] = total_duration
    results["all_passed"] = minimax_passed and datagen_passed

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    minimax_count = len(minimax_tester.results)
    minimax_pass = sum(1 for r in minimax_tester.results if r.passed)
    datagen_count = len(datagen_tester.results)
    datagen_pass = sum(1 for r in datagen_tester.results if r.passed)

    total_tests = minimax_count + datagen_count
    total_pass = minimax_pass + datagen_pass

    print(f"\nGPU Minimax:        {minimax_pass}/{minimax_count} tests passed")
    print(f"GPU Data Generation: {datagen_pass}/{datagen_count} tests passed")
    print(f"---")
    print(f"Total:              {total_pass}/{total_tests} tests passed")
    print(f"Duration:           {total_duration:.1f}s")

    if results["all_passed"]:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")

    return results


def main():
    parser = argparse.ArgumentParser(description="RingRift GPU Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks only")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--output-json", type=str, help="Output results to JSON file")

    args = parser.parse_args()

    # Determine mode
    if args.benchmark:
        mode = "benchmark"
    elif args.full:
        mode = "full"
    else:
        mode = "quick"

    # Run tests
    results = run_all_tests(mode=mode, verbose=not args.quiet)

    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")

    # Exit with appropriate code
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
