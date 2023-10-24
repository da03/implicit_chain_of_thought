export MODEL=gpt2-medium
export EPOCHS=120
export LR=5e-5
export BSZ=32
export FOLDER=long_mult_4_inter
export MODELSAVE="${MODEL////_}"
export SAVE=long_mult_4_baselines/nocot/${MODEL}/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
export MODEL=gpt2-large
export EPOCHS=120
export LR=5e-5
export BSZ=32
export FOLDER=long_mult_4_inter
export MODELSAVE="${MODEL////_}"
export SAVE=long_mult_4_baselines/nocot/${MODEL}/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
export MODEL=gpt2-medium
export EPOCHS=120
export LR=5e-5
export BSZ=32
export FOLDER=long_mult_5_inter
export MODELSAVE="${MODEL////_}"
export SAVE=long_mult_5_baselines/nocot/${MODEL}/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
export MODEL=gpt2-large
export EPOCHS=120
export LR=5e-5
export BSZ=32
export FOLDER=long_mult_5_inter
export MODELSAVE="${MODEL////_}"
export SAVE=long_mult_5_baselines/nocot/${MODEL}/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
export MODEL=gpt2-large
export EPOCHS=120
export LR=5e-5
export BSZ=32
export FOLDER=long_mult_5_inter
export MODELSAVE="${MODEL////_}"
export SAVE=long_mult_5_baselines/nocot/${MODEL}/e${EPOCHS}_cont
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_baselines/nocot/gpt2-large/e120/checkpoint_31_5e-05_gpt2-large \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
