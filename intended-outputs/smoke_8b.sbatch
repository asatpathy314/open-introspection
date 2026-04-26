#!/bin/bash
#SBATCH -J intended_outputs_8b_smoke
#SBATCH -p interactive
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -t 0:45:00
#SBATCH -o /sfs/gpfs/tardis/home/cqa3ym/Experiments/open-introspection/intended-outputs/smoke_8b.%j.out
#SBATCH -e /sfs/gpfs/tardis/home/cqa3ym/Experiments/open-introspection/intended-outputs/smoke_8b.%j.err

set -euo pipefail
cd /sfs/gpfs/tardis/home/cqa3ym/Experiments/open-introspection
source .venv/bin/activate

echo "=== env ==="
nvidia-smi
echo "=== start: $(date) ==="

python intended-outputs/run_intended_outputs.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --vector-dir data/vectors/llama-3.1-8b-instruct \
    --results-dir data/results/intended-outputs-8b-smoke \
    --layers 15 \
    --num-repetitions 1 \
    --strengths 2 \
    --results-file smoke.jsonl

echo "=== end: $(date) ==="
