import argparse
import glob
import os
from typing import Optional, Sequence

from .config import load_method_eval_config
from .runner import merge_method_summaries, run_evaluation


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run uniGradICON per-method test-set evaluation or merge method summaries."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Single-method evaluation YAML config.",
    )
    group.add_argument(
        "--merge_glob",
        type=str,
        help="Glob pattern for summary CSV files to merge (e.g., results/eval/*/summary.csv).",
    )
    parser.add_argument(
        "--merge_out",
        type=str,
        default="results/eval/final_summary.csv",
        help="Output CSV path for merged summaries.",
    )
    args = parser.parse_args(argv)

    if args.config:
        settings = load_method_eval_config(args.config)
        output_dir = run_evaluation(settings)
        print(f"Evaluation complete. Outputs written to: {output_dir}")
        return

    summary_paths = sorted(glob.glob(args.merge_glob))
    if not summary_paths:
        raise FileNotFoundError(f"No summary files matched pattern: {args.merge_glob}")
    merge_out = os.path.abspath(args.merge_out)
    merge_method_summaries(summary_paths, merge_out)
    print(f"Merged {len(summary_paths)} summaries into: {merge_out}")
