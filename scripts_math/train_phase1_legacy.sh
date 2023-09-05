export EPOCHS=60
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=math_${D}
export CKPTEPOCH=14
export QMODEL=math_${D}_cot_${MODEL}/checkpoint_14_5e-05_gpt2
export AMODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export MODELSAVE="${MODEL////_}"
export SAVE=model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math.py \
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
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=60
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=math_${D}
export CKPTEPOCH=14
export QMODEL=math_${D}_cot_${MODEL}/checkpoint_14_5e-05_gpt2
export AMODEL=model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e15_f${F}_opt/checkpoint_${CKPTEPOCH}_5e-05
export MODELSAVE="${MODEL////_}"
export SAVE=model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math.py \
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
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
