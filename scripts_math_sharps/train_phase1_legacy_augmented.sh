# interval 0

export EPOCHS=10
export INTERVAL=0
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=7
export QMODEL=augmented_math_${D}_cot_${MODEL}_repharvard/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model $SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


# interval 2

export EPOCHS=10
export INTERVAL=2
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=7
export QMODEL=augmented_math_${D}_cot_${MODEL}_repharvard/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model $SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
