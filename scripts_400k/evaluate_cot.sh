export FOLDER=nobrackets_400kaugmented_math_scaffolding_formula
export SAVE=evalulations/400k/baselines/cot/gptsmall
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python evaluate_cot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptsmall/checkpoint_12_5e-05_gpt2 \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &

export FOLDER=nobrackets_400kaugmented_math_scaffolding_formula
export SAVE=evalulations/400k/baselines/cot/gptmedium
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python evaluate_cot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export FOLDER=nobrackets_400kaugmented_math_scaffolding_formula
export SAVE=evalulations/400k/baselines/cot/gptlarge
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python evaluate_cot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptlarge/checkpoint_4_5e-05_gpt2-large \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
