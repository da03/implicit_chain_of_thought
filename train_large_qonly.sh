export EPOCHS=15
export LR=1e-5
export FOLDER=math_scaffolding_formula
export MODEL=gpt2-large
CUDA_VISIBLE_DEVICES=0 python train_large_qonly.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    > log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR} 2>&1&
