export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_cot.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    > mult_cot/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=1e-5
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_cot.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    > mult_cot/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=3e-4
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_cot.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    > mult_cot/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=1e-4
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_cot.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    > mult_cot/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
