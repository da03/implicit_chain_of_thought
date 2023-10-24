export MODEL=gpt2
export EPOCHS=15
export LR=5e-5
export BSZ=32
export FOLDER=nobrackets_400kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=400k_baselines/cot_nobrackets/gptsmall
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
export MODEL=gpt2-medium
export EPOCHS=15
export LR=5e-5
export BSZ=32
export FOLDER=nobrackets_400kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=400k_baselines/cot_nobrackets/gptmedium
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
export MODEL=gpt2-large
export EPOCHS=15
export LR=5e-5
export BSZ=32
export FOLDER=nobrackets_400kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=400k_baselines/cot_nobrackets/gptlarge
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
