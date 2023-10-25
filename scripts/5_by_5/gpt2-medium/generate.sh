
export FOLDER=/n/holyscratch01/rush_lab/Users/yuntian/implicit/final_release/data/5_by_5_mult
export STUDENT=/n/holyscratch01/rush_lab/Users/yuntian/implicit/final_release/models/5_by_5_mult/gpt2-medium/student
export EMULATOR=/n/holyscratch01/rush_lab/Users/yuntian/implicit/final_release/models/5_by_5_mult/gpt2-medium/emulator
export BSZ=1
export SAVE=
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --batch_size $BSZ \
    --test_path data/${FOLDER}/test_bigbench.txt \
    --student_path $STUDENT \
    --emulator_path $EMULATOR \
    > ${SAVE}/log.generate 2>&1&

# no_mixture should default to 1
