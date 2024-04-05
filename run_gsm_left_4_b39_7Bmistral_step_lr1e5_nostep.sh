#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 60000 # memory pool for all cores
#SBATCH -t 6-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
export BASE=/n/shieber_lab/Lab/yuntian/cascade/implicit_chain_of_thought
cd $BASE
conda activate /n/shieber_lab/Lab/yuntian/icml/conda_icml
which python


export FOLDER=data/gsm8k
export MODEL=mistralai/Mistral-7B-v0.1
export EPOCHS=40
export LR=1e-5
export BSZ=16
export A=2
export SIDE=left
export PERE=4
export TYPE=epoch
export BEYOND=39
export MAXLENT=256
export M=${MODEL#*/}
export SAVE=train_models/gsm8k/gpt2/teacher_s${SIDE}_p${PERE}_b${BEYOND}_e${EPOCHS}_m${M}_bsz${BSZ}_a${A}_t${TYPE}_maxlent${MAXLENT}_lr1e5
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_curriculum_learning.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/test.txt \
    --epochs $EPOCHS \
    --delete_per_epoch $PERE \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --delete_side $SIDE \
    --delete_beyond $BEYOND \
    --delete_type $TYPE \
    --max_len_train $MAXLENT \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit_cascade_feb24/$SAVE \
    > ${SAVE}/log.train 2>&1
