export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export SAVE=evalulations/300k/baselines/nocot/gptsmall
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/300k_baselines/nocot/gptsmall/e60/checkpoint_4_5e-05_gpt2 \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export SAVE=evalulations/300k/baselines/nocot/gptmedium
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/300k_baselines/nocot/gptmedium/e60/checkpoint_4_5e-05_gpt2-medium \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &


export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export SAVE=evalulations/000k/baselines/nocot/gptlarge
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_nocot_savemodel_math.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/200k_baselines/nocot/gptlarge/e60/checkpoint_2_5e-05_gpt2-large \
    --batch_size 1 \
    --compile 0 \
    > ${SAVE}/log.train.text.model.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1 &
