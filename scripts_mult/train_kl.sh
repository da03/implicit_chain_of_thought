export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-medium
export D=4
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptmedium
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_kl_anneal.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-large
export D=4
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_0_5e-05_gpt2-large
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptlarge_ckpt0
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_kl_anneal.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&








export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-medium
export D=4
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptmedium_anneal5
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_kl_anneal_10.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-large
export D=4
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_0_5e-05_gpt2-large
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptlarge_ckpt0_anneal5
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_kl_anneal_10.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-medium
export D=4
export QMODEL=gpt2-medium
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptmedium_anneal5_scratch
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_kl_anneal_10.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-large
export D=4
export QMODEL=gpt2-large
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptlarge_ckpt0_anneal10_linear
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_kl_anneal_10_linear.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export MODEL=gpt2-large
export D=4
export QMODEL=gpt2-large
export FOLDER=long_mult_${D}_inter
export BSZ=32
export SAVE=model_kl_${D}_inter_gptlarge_ckpt0_anneal10
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_kl_anneal_10_real10.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
