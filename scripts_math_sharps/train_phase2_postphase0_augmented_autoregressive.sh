# interval 0
export INTERVAL=0
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=2
export QCKPTEPOCH=9
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPT=7
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
export MODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e10_legacy_2_mgpt2_try1_7_repharvard_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest_interval${INTERVAL}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive.py \
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


# interval 2
export INTERVAL=2
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=2
export QCKPTEPOCH=2
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPT=7
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
export MODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e10_legacy_2_mgpt2_try1_7_repharvard_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest_interval${INTERVAL}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive.py \
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


# interval 1
export INTERVAL=1
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=2
export QCKPTEPOCH=2
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPT=7
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
export MODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e10_legacy_${CKPTEPOCH}_mgpt2_try1_7_repharvard_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest_interval${INTERVAL}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive.py \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
# interval 3
export INTERVAL=3
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=3
export QCKPTEPOCH=3
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPT=7
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
export MODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e10_legacy_${CKPTEPOCH}_mgpt2_try1_7_repharvard_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest_interval${INTERVAL}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive.py \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
# interval 4
export INTERVAL=4
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export CKPTEPOCH=3
export QCKPTEPOCH=3
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPT=7
export MODEL=gpt2
export MODELSAVE="${MODEL////_}"
export MODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e10_legacy_${CKPTEPOCH}_mgpt2_try1_7_repharvard_interval${INTERVAL}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_fixed_model_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_try1_legacybest_interval${INTERVAL}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive.py \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
