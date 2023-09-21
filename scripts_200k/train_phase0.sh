export INTERVAL=4
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
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
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f$F
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
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
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=3
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row_above
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=3
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_interval${INTERVAL}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2-medium
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=3
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=5
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2-medium
export MODELSAVE="${MODEL////_}"
export SAVE=gpt2medium_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_interval${INTERVAL}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=3
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=1
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=residual_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=1
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=residual_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&



export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=mlp_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=mlp_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt.train \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_augmented_math_${D}
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=mlp_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_fromscratch
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_fromscratch.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt.train \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=1
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k_residualmlp_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=1
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k_residualmlp_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_noresidual
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k_minus_phase0_rnn/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_noresidual
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_rnn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2-medium
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=5
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2-medium
export MODELSAVE="${MODEL////_}"
export SAVE=200k_gptmedium_minus_phase0_rnn/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_noresidual
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_rnn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=bottom_row
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_noresidual
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_noresidual
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=0
echo $INTERVAL
export D=scaffolding_formula
export MODEL=gpt2
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k_minus_phase0/sharps_augmented_model_${D}_phase0_inter_fixq_qgpt2_p${MODELSAVE}_r${R}_m${MODE}_e${EPOCHS}_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k_f${F}_minus${MINUS}_normgrad_noresidual
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&




### Sep 18
# phase 0, no rnn, gptsmall, interval 0
export INTERVAL=0
export MODEL=gpt2
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export MODELSAVE="${MODEL////_}"
export SAVE=200k/phase0/interval${INTERVAL}/nornn/gptsmall/r${R}_m${MODE}_e${EPOCHS}_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

# phase 0, no rnn, gptmedium, interval 0
export INTERVAL=0
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export MODELSAVE="${MODEL////_}"
export SAVE=200k/phase0/interval${INTERVAL}/nornn/gptmedium/r${R}_m${MODE}_e${EPOCHS}_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --last_id_minus $MINUS \
    --mode $MODE \
    --residual $R \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --interval $INTERVAL \
    --compile 0 \
    --max_new_tokens 10 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
