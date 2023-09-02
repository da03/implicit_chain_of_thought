export D=5
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=6
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=4
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=5
export MODEL=gpt2
export F=top_row
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=6
export MODEL=gpt2
export F=top_row
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=5
export MODEL=gpt2
export F=first_column
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=5
export MODEL=gpt2
export F=bottom_row
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=5
export MODEL=gpt2
export F=last_column
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=4
export MODEL=gpt2
export F=first_column
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=7
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=8
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=9
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=5
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2medium_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=5
export MODEL=gpt2-large
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2large_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&





export D=5
export MODEL=gpt2
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try2
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=5
export MODEL=gpt2
export F=diagonal_orig
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=5
export MODEL=gpt2-medium
export F=diagonal_orig
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2medium_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=5
export MODEL=gpt2-large
export F=diagonal_orig
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2large_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=6
export MODEL=gpt2
export F=diagonal_orig
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=5
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2medium_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export D=5
export MODEL=gpt2-large
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2large_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&



export D=6
export MODEL=gpt2-medium
export F=diagonal_orig
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2medium_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=6
export MODEL=gpt2-large
export F=diagonal_orig
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2large_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export D=6
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptmedium/checkpoint_1_5e-05_gpt2-medium
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2medium_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export D=6
export MODEL=gpt2-large
export F=diagonal
export EPOCHS=120
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=long_mult_${D}_inter
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_cot_${D}_inter_gptlarge/checkpoint_1_5e-05_gpt2-large
#export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/model_cot_${D}_inter/checkpoint_1_5e-05_gpt2
export SAVE=model_phase0_${D}_inter_fixq_qgpt2large_p${MODEL}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
