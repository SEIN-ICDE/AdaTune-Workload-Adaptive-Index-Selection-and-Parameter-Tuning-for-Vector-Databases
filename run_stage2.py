"""
run_stage2.py  —  AdaTune Stage 2 entry point

Requires Stage 1 to have been run first for the same dataset/threshold.

Usage:
  python run_stage2.py --dataset gist     --threshold 0.90
  python run_stage2.py --dataset sift     --threshold 0.95
  python run_stage2.py --dataset glove    --threshold 0.90
  python run_stage2.py --dataset wikipedia --threshold 0.75
"""

import argparse

from adatune.stage2 import run_stage2

CONFIGS = {}
from configs.gist      import CFG as GIST_CFG;      CONFIGS["gist"]      = GIST_CFG
from configs.sift      import CFG as SIFT_CFG;      CONFIGS["sift"]      = SIFT_CFG
from configs.glove     import CFG as GLOVE_CFG;     CONFIGS["glove"]     = GLOVE_CFG
from configs.wikipedia import CFG as WIKI_CFG;      CONFIGS["wikipedia"] = WIKI_CFG

VALID_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaTune Stage 2")
    parser.add_argument("--dataset",   required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--threshold", required=True, type=float,
                        choices=VALID_THRESHOLDS,
                        help="Target recall threshold")
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    run_stage2(cfg, args.threshold)
