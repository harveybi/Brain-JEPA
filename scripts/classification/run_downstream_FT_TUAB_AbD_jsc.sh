#!/bin/bash
#SBATCH --account=hai_1276
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=/p/project1/hai_1148/results/logs/TUAB/%j.out
#SBATCH --error=/p/project1/hai_1148/results/logs/TUAB/%j.err

set -euo pipefail

SEED="${1:-}"
BLR="${2:-}"
WEIGHT_DECAY="${3:-}"
LAYER_DECAY="${4:-}"
if [[ -z "${SEED}" || -z "${BLR}" || -z "${WEIGHT_DECAY}" || -z "${LAYER_DECAY}" ]]; then
    echo "Usage: sbatch $0 <seed> <blr> <weight_decay> <layer_decay>"
    exit 1
fi

REPO_ROOT="/p/project1/hai_1148/Brain-JEPA"
DATA_ROOT="${REPO_ROOT}/data"
OUTPUT_ROOT="${REPO_ROOT}/output_dirs"
LOG_ROOT="/p/project1/hai_1148/results/logs/TUAB"
PRETRAIN_CKPT="${REPO_ROOT}/path/to/jepa-ep300.pth.tar"

mkdir -p "${LOG_ROOT}"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Set PRETRAIN_CKPT in $0 before submitting. Missing: ${PRETRAIN_CKPT}"
    exit 1
fi

cd "${REPO_ROOT}"
source /p/project1/hai_1024/bi1/miniforge3/etc/profile.d/conda.sh
conda activate brain-jepa

python downstream_eval.py \
    --config "${REPO_ROOT}/configs/downstream/fine_tune.yaml" \
    --downstream_task fine_tune \
    --batch_size 16 \
    --epochs 50 \
    --blr "${BLR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --layer_decay "${LAYER_DECAY}" \
    --min_lr 0.000001 \
    --smoothing 0.0 \
    --seed "${SEED}" \
    --output_root "${OUTPUT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --model_name vit_base \
    --data_make_fn TUAB \
    --load_path "${PRETRAIN_CKPT}" \
    --use_normalization \
    --crop_size 400,160 \
    --patch_size 16 \
    --pred_depth 12 \
    --pred_emb_dim 384 \
    --attn_mode flash_attn \
    --add_w mapping \
    --gradient_csv "${DATA_ROOT}/gradient_mapping_400.csv"
