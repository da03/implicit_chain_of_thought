export EPOCHS=40
export LR=5e-5
export D=5
export CKPTEPOCH=19
export QCKPTEPOCH=23
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_5_inter_fixq_qgpt2large_pgpt2-large_r0_mbottom_e120_fdiagonal_opt_try1/checkpoint_${CKPTEPOCH}_5e-05
#/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_gpt2large_fixed_e60_nolegacy_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2large${CKPTEPOCH}_qgpt2large${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

