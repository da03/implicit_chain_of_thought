export EPOCHS=15
export LR=5e-5
export D=scaffolding_formula
export QCKPTEPOCH=3
export FOLDER=augmented_math_${D}
export F=diagonal
export CKPT=5
export CKPTEPOCH=3
export MODEL=augmented_model_${D}_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt_${CKPT}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_phase1_scaffolding_formula_inter_Fdiagonal_gpt2large_fixed_e60_qe5_nolegacy_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=augmented_model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1_nolegacybest
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
    --compile 0 \
    --save_model $SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=scaffolding_formula
export QCKPTEPOCH=3
export FOLDER=augmented_math_${D}
export F=diagonal
export CKPT=5
export CKPTEPOCH=3
export MODEL=augmented_model_${D}_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt_${CKPT}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_phase1_scaffolding_formula_inter_Fdiagonal_gpt2large_fixed_e60_qe5_nolegacy_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=augmented_model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1_nolegacybest
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
    --compile 0 \
    --save_model $SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=scaffolding_formula
export QCKPTEPOCH=59
export FOLDER=augmented_math_${D}
export F=diagonal
export CKPT=5
export CKPTEPOCH=3
export MODEL=augmented_model_${D}_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt_${CKPT}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_phase1_scaffolding_formula_inter_Fdiagonal_gpt2large_fixed_e60_qe5_nolegacy_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=augmented_model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1_nolegacylast
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
    --compile 0 \
    --save_model $SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
