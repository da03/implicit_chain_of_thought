export EPOCHS=60
export D=5
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptxl/checkpoint_1_5e-05_gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2_fixed_e${EPOCHS}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=60
export D=5
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=top_row
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptxl/checkpoint_1_5e-05_gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2_fixed_e${EPOCHS}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=60
export D=5
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptxl/checkpoint_1_5e-05_gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2_fixed_e${EPOCHS}_nolegacy
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=60
export D=5
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptxl/checkpoint_1_5e-05_gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=60
export D=5
export MODEL=gpt2-medium
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2medium_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=60
export D=5
export MODEL=gpt2-large
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2large_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&



export EPOCHS=60
export D=5
export MODEL=gpt2-medium
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2medium_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=60
export D=5
export MODEL=gpt2-large
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2large_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=60
export D=6
export MODEL=gpt2-medium
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2medium_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=60
export D=6
export MODEL=gpt2-large
export LR=5e-5
export BSZ=32
export F=diagonal_orig
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
#export FMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase1_${D}_inter_F${F}_gpt2large_fixed_e${EPOCHS}_nolegacy_try1
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
