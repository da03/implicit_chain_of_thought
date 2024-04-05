#!/bin/bash
#SBATCH -p seas_gpu # partition (queue)
#SBATCH -n 1 # number of cores
#SBATCH --mem 60000 # memory pool for all cores
#SBATCH -t 7-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
export BASE=/n/shieber_lab/Lab/yuntian/cascade/implicit_chain_of_thought
cd $BASE
conda activate /n/shieber_lab/Lab/yuntian/icml/conda_icml
which python


export FOLDER=/n/rush_lab/Lab/Users/yuntian/implicit/data/long_mult_12_inter_mar2124_short_binarytree
export MODEL=gpt2
export EPOCHS=200
export LR=5e-5
export BSZ=32
export SIDE=left
export PERE=8
export BEYOND=-1
export TYPE=step
export LAMB=0.1
export PRETRAIN=0
export SAVE=train_models/shiftall_binarytree_12_by_12_mult/gpt2/teacher_s${SIDE}_p${PERE}_b${BEYOND}_e${EPOCHS}_t${TYPE}_l${LAMB}_or_accuracy1_pretrain${PRETRAIN}_keepall
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_curriculum_learning_lowdata_distribution_expand_position_keeppos_or_accuracy1.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --delete_per_epoch $PERE \
    --pretrain_epoch $PRETRAIN \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --delete_side $SIDE \
    --delete_beyond $BEYOND \
    --delete_type $TYPE \
    --lamb $LAMB \
    --reset_optimizer \
    --keep_allposition \
    --oraccuracy \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit_cascade_mar18/$SAVE \
    > ${SAVE}/log.train 2>&1
