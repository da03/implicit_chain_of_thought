
### 1. Prepare data by concatenating two equations together to form one example
python twoproblems/convert_data.py data/4_by_4_mult data/4_by_4_mult_twoproblems

### 2. Train a teacher (on one equation only)
export FOLDER=data/4_by_4_mult_twoproblems
export MODEL=gpt2-medium
export EPOCHS=1
export LR=5e-5
export BSZ=16
export SAVE=train_models/4_by_4_mult_twoproblems/gpt2-medium/teacher
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --save_model $SAVE \
    > ${SAVE}/log.train 2>&1&

### 3. Train a mind reading student (input: added teacher states on two equations and the entire example with two equations concatenated)
export FOLDER=data/4_by_4_mult_twoproblems
export DELTA=dynamic
export MODEL=gpt2-medium
export EPOCHS=40
export LR=5e-5
export BSZ=32
export TEACHER=train_models/4_by_4_mult_twoproblems/gpt2-medium/teacher/checkpoint_0
export SAVE=train_models/4_by_4_mult_twoproblems/gpt2-medium/student_initial
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_mind_reading_student.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --teacher $TEACHER \
    --save_model $SAVE \
    --delta $DELTA \
    > ${SAVE}/log.train 2>&1&

### 4. Train a thought emulator (emulating added teacher states on two equations)
export FOLDER=data/4_by_4_mult_twoproblems
export DELTA=dynamic
export MODEL=gpt2-medium
export EPOCHS=40
export LR=5e-5
export BSZ=32
export MIXTURE_SIZE=1
export TEACHER=train_models/4_by_4_mult_twoproblems/gpt2-medium/teacher/checkpoint_0
export SAVE=train_models/4_by_4_mult_twoproblems/gpt2-medium/emulator_initial
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python src/train_thought_emulator.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --base_model $MODEL \
    --teacher $TEACHER \
    --save_model $SAVE \
    --delta $DELTA \
    --mixture_size ${MIXTURE_SIZE} \
    > ${SAVE}/log.train 2>&1&

### 5. Train coupled system
export FOLDER=data/4_by_4_mult_twoproblems
export EPOCHS=40
export LR=5e-5
export BSZ=32
export STUDENT=train_models/4_by_4_mult_twoproblems/gpt2-medium/student_initial/checkpoint_39
export EMULATOR=train_models/4_by_4_mult_twoproblems/gpt2-medium/emulator_initial/checkpoint_39
export SAVE=train_models/4_by_4_mult_twoproblems/gpt2-medium/
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_coupled_emulator_and_student.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BSZ \
    --student $STUDENT \
    --emulator $EMULATOR \
    --save_model $SAVE \
    > ${SAVE}/log.train 2>&1&
