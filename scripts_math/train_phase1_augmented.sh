export EPOCHS=60
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export QCKPTEPOCH=5
export QMODEL=augmented_math_${D}_cot_${MODELSAVE}/checkpoint_${QCKPTEPOCH}_5e-05_gpt2
export SAVE=augmented_model_phase1_${D}_inter_F${F}_gpt2large_fixed_e${EPOCHS}_qe${QCKPTEPOCH}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --compile 0 \
    --save_model $SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
