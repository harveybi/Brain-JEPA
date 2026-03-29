#!/bin/bash
#SBATCH --account=hai_1276
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=develbooster
#SBATCH --cpus-per-task=48
#SBATCH --output=/p/project1/hai_1148/results/logs/ADNI/%j.out
#SBATCH --error=/p/project1/hai_1148/results/logs/ADNI/%j.err
#SBATCH --time=00:20:00

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=12

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells on JSC machines.
MASTER_ADDR="$MASTER_ADDR"i

export MASTER_PORT=29500

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# source /p/project1/fomo/guo3/miniconda3/bin/activate brainharmonix
source /p/project1/hai_1024/bi1/miniforge3/etc/profile.d/conda.sh
conda activate brain-jepa


srun python downstream_eval.py \
    --downstream_task fine_tune \
    --task classification \
    --batch_size 16 \
    --nb_classes 2 \
    --num_seed 5 \
    --load_epoch 300 \
    --epochs 50 \
    --blr 0.001 \
    --min_lr 0.000001 \
    --smoothing 0.0 \
    --config /p/project1/hai_1024/Brain-JEPA/configs/downstream/fine_tune.yaml \
    --output_root /p/project1/hai_1024/Brain-JEPA/output_dirs \
    --model_name vit_base \
    --data_make_fn hca_sex \
    --load_path /p/project1/hai_1024/Brain-JEPA/logs/Pretraining \
    --use_normalization \
    --crop_size 450,160 \
    --patch_size 16 \
    --pred_depth 12 \
    --pred_emb_dim 384 \
    --attn_mode flash_attn \
    --add_w mapping \
    --downsample 
