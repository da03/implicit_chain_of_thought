export EPOCHS=15
export LR=5e-5
export MODEL=gpt2
export BSZ=32
export D=4
export F=top_row
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase1.py \
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
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export MODEL=gpt2
export BSZ=32
export D=4
export F=bottom_row
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase1.py \
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
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export MODEL=gpt2
export BSZ=32
export D=4
export F=last_column
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase1.py \
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
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export MODEL=gpt2
export BSZ=32
export D=4
export F=diagonal
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase1.py \
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
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
