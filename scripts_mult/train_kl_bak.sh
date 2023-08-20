export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_mult_qonly_diag_kl.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > mult_kl_postfix_interleave_gpt2_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mult_qonly_diag_kl_anneal.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > mult_kl_postfix_interleave_gpt2_multmlp_anneal/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
