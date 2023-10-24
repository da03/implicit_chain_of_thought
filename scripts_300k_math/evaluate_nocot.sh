export FOLDER=long_mult_4_inter
export MODELSAVE="${MODEL////_}"
export SAVE=evalulations/mult/4/gptsmall
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --model /n/rush_lab/Lab/Users/yuntian/implicit/model_nocot_4/checkpoint_14_5e-05_gpt2 \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &

export FOLDER=long_mult_5_inter
export MODELSAVE="${MODEL////_}"
export SAVE=evalulations/mult/5/gptsmall
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --model /n/rush_lab/Lab/Users/yuntian/implicit/model_nocot_5/checkpoint_14_5e-05_gpt2 \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export FOLDER=long_mult_4_inter
export MODELSAVE="${MODEL////_}"
export SAVE=evalulations/mult/4/gptmedium
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/model_nocot_4_gptmedium/checkpoint_14_5e-05_gpt2-medium \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &

export FOLDER=long_mult_5_inter
export MODELSAVE="${MODEL////_}"
export SAVE=evalulations/mult/5/gptmedium
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/model_nocot_5_gptmedium/checkpoint_14_5e-05_gpt2-medium \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export FOLDER=long_mult_4_inter
export MODELSAVE="${MODEL////_}"
export SAVE=evalulations/mult/4/gptlarge
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/model_nocot_4_gptlarge/checkpoint_14_5e-05_gpt2-large \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &

export FOLDER=long_mult_5_inter
export MODELSAVE="${MODEL////_}"
export SAVE=evalulations/mult/5/gptlarge
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/model_nocot_5_gptlarge/checkpoint_14_5e-05_gpt2-large \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
