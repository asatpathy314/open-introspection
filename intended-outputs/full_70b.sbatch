#!/bin/bash
#SBATCH -J intended_outputs_70b
#SBATCH -p gpu-a100-80
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=360G
#SBATCH -c 16
#SBATCH -t 3-00:00:00
#SBATCH -o /sfs/gpfs/tardis/home/cqa3ym/Experiments/open-introspection/intended-outputs/full_70b.%j.out
#SBATCH -e /sfs/gpfs/tardis/home/cqa3ym/Experiments/open-introspection/intended-outputs/full_70b.%j.err

set -euo pipefail
cd /sfs/gpfs/tardis/home/cqa3ym/Experiments/open-introspection
source .venv/bin/activate

set -a
source .env
set +a

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_CACHE=/scratch/$USER/hf_cache/hub
export UV_CACHE_DIR=/scratch/$USER/uv_cache
mkdir -p "$HF_HUB_CACHE" "$UV_CACHE_DIR"

echo "=== env ==="
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
nvidia-smi
echo "=== start: $(date) ==="

python intended-outputs/run_intended_outputs.py \
    --model-id meta-llama/Llama-3.1-70B-Instruct \
    --vector-dir data/vectors/llama-3.1-70b-instruct \
    --results-dir data/results/intended-outputs-llama31-70b \
    --results-file results.jsonl

echo "=== end: $(date) ==="
