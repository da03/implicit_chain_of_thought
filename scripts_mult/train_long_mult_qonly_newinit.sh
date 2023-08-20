export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mult_qonly.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_qonly_newinit/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
