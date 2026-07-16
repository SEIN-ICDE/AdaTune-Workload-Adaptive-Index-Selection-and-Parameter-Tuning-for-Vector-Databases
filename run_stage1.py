"""
run_stage1.py  —  AdaTune Stage 1 entry point

Usage:
  python run_stage1.py --dataset gist     --threshold 0.90
  python run_stage1.py --dataset sift     --threshold 0.95
  python run_stage1.py --dataset glove    --threshold 0.90
  python run_stage1.py --dataset wikipedia --threshold 0.75
"""

import argparse

from adatune.stage1 import run_stage1

CONFIGS = {}
from configs.gist      import CFG as GIST_CFG;      CONFIGS["gist"]      = GIST_CFG
from configs.sift      import CFG as SIFT_CFG;      CONFIGS["sift"]      = SIFT_CFG
from configs.glove     import CFG as GLOVE_CFG;     CONFIGS["glove"]     = GLOVE_CFG
from configs.wikipedia import CFG as WIKI_CFG;      CONFIGS["wikipedia"] = WIKI_CFG

VALID_THRESHOLDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaTune Stage 1")
    parser.add_argument("--dataset",   required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--threshold", required=True, type=float,
                        choices=VALID_THRESHOLDS,
                        help="Target recall threshold")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    run_stage1(cfg, args.threshold, milvus_uri=args.milvus_uri)
