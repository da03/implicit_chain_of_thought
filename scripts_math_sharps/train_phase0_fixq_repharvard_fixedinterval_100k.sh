for INTERVAL in 4
do
    echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
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
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
done
