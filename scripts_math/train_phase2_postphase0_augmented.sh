export EPOCHS=40
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=14
export QCKPTEPOCH=59
export FOLDER=augmented_math_${D}
export F=diagonal
export MODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_scaffolding_formula_phase1_inter_F${F}_gpt2_fixed_e60_legacy_14_mgpt2_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export MODELSAVE="${MODEL////_}"
export SAVE=augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model $SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=40
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=14
export QCKPTEPOCH=3
export FOLDER=math_${D}
export F=diagonal_orig
export MODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=model_scaffolding_formula_phase1_inter_F${F}_gpt2_fixed_e60_legacy_14_mgpt2_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export MODELSAVE="${MODEL////_}"
export SAVE=model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1
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
    --save_model $SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

# legacy best
export EPOCHS=40
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=3
export QCKPTEPOCH=2
export FOLDER=augmented_math_${D}
export F=diagonal
export CKPT=5
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
#export MODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export MODEL=augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r0_mbottom_e15_f${F}_opt_${CKPT}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e60_legacy_3_mgpt2_try1_5/checkpoint_2_5e-05
export BSZ=32
export A=1
export SAVE=augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model $SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


# legacy last
export EPOCHS=40
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=3
export QCKPTEPOCH=2
export FOLDER=augmented_math_${D}
export F=diagonal
export CKPT=5
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
#export MODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export MODEL=augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r0_mbottom_e15_f${F}_opt_${CKPT}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e60_legacy_3_mgpt2_try1_5/checkpoint_59_5e-05
export BSZ=32
export A=1
export SAVE=augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacylast
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model $SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


# legacy best
export EPOCHS=15
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=3
export QCKPTEPOCH=2
export FOLDER=augmented_math_${D}
export F=diagonal
export CKPT=5
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
#export MODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export CKPTEPOCH=3
export CKPT=7
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt_${CKPT}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e40_legacy_3_mgpt2_try1_7_legacy/checkpoint_2_5e-05
export BSZ=32
export A=1
export SAVE=augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest_saveshieber
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math.py \
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
