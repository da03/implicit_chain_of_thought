export MODEL=gpt2
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=60
export LR=5e-5
export BSZ=32
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=200k_baselines/nocot/gptsmall/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export MODEL=gpt2-medium
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=60
export LR=5e-5
export BSZ=32
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=200k_baselines/nocot/gptmedium/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export MODEL=gpt2-large
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=60
export LR=5e-5
export BSZ=32
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=200k_baselines/nocot/gptlarge/e${EPOCHS}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_nocot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


