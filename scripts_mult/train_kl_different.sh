export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-large
export D=4
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export FOLDER=long_mult_${D}_inter
export BSZ=32
export R=0
export MODE=bottom
export SAVE=model_kl_different_${D}_inter_qgpt2medium_pgpt2large_r${R}_m${MODE}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_kl_anneal_10_real10_different.py \
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
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
