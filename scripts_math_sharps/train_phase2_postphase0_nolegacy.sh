# interval 0
export INTERVAL=0
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=3
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_phase1_scaffolding_formula_inter_Fdiagonal_gpt2large_fixed_e30_qe6_nolegacy_try1_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=sharps_augmented_model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1_nolegacybest_inter${INTERVAL}_repharvard_100k
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

# interval 2
export INTERVAL=2
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=3
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_phase1_scaffolding_formula_inter_Fdiagonal_gpt2large_fixed_e30_qe6_nolegacy_try1_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=sharps_augmented_model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1_nolegacybest_inter${INTERVAL}_repharvard_100k
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
