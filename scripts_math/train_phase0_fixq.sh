export D=scaffolding_formula
export MODEL=gpt2
export F=diagonal
export EPOCHS=15
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=augmented_math_${D}
export CKPT=7
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=scaffolding_formula
export MODEL=gpt2
export F=diagonal_orig
export EPOCHS=15
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=augmented_math_${D}
export CKPT=7
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
