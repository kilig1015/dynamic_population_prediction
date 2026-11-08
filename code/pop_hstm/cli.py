from __future__ import annotations

import sys
from typing import Optional, Sequence


def main_entry(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--benchmark-runtime" in args:
        from .benchmark import run_runtime_benchmark

        benchmark_argv = [arg for arg in args if arg != "--benchmark-runtime"]
        run_runtime_benchmark(benchmark_argv)
    elif "--mode-rs" in args:
        from .random_search import run_random_search_mode

        rs_argv = [arg for arg in args if arg != "--mode-rs"]
        run_random_search_mode(rs_argv)
    else:
        from .workflow import run_standard_workflow

        run_standard_workflow()
