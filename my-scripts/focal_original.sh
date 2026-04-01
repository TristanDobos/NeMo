#!/bin/bash
#$ -o /mnt/matylda6/xdobos00/runs/logs/focalcodec-$JOB_NAME.$JOB_ID.out
#$ -e /mnt/matylda6/xdobos00/runs/logs/focalcodec-$JOB_NAME.$JOB_ID.err

# 1. Initialize Conda for this subshell
source /mnt/matylda6/xdobos00/miniconda/etc/profile.d/conda.sh

# 2. Activate your target environment
# (Remove the . /.../bin/activate line, it conflicts with conda)
conda activate /mnt/matylda6/xdobos00/focal-pv310

# 3. Fix the Allocation Warning
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NEMO_DISABLE_ONE_LOGGER=1
export WANDB_MODE=offline

# 4. GPU Selection
export CUDA_VISIBLE_DEVICES=$(~/scripts/free-gpus.sh 1)

# 5. Run Python
echo "Running: python /mnt/matylda6/xdobos00/focal-original/focalcodec/focalcodec/try1.py"
python /mnt/matylda6/xdobos00/focal-original/focalcodec/focalcodec/try1.py