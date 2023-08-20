export EPOCHS=15
export LR=5e-5
export D=4
export FOLDER=long_mult_${D}_inter
export MODEL=gpt2-medium
export CKPTEPOCH=3
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_gptmedium/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_gptmedium
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export FOLDER=long_mult_${D}_inter
export MODEL=gpt2-medium
export CKPTEPOCH=5
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_gptmedium/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_gptmedium
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export D=4
export FOLDER=long_mult_${D}_inter
export MODEL=gpt2-medium
export CKPTEPOCH=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_gptmedium/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_gptmedium
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
