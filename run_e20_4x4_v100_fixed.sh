#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 60000 # memory pool for all cores
#SBATCH -t 6-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1


export BASE=/n/shieber_lab/Lab/yuntian/cascade/implicit_chain_of_thought
cd $BASE
conda activate /n/shieber_lab/Lab/yuntian/icml/conda_icml
which python
export FOLDER=data/4_by_4_mult
export MODEL=gpt2
export EPOCHS=20
export LR=5e-5
export BSZ=32
export SAVE=train_models/4_by_4_mult/gpt2/teacher_e${EPOCHS}_v100_fix2eos
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_curriculum_learning.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit_cascade_feb14/$SAVE \
    > ${SAVE}/log.train 2>&1
