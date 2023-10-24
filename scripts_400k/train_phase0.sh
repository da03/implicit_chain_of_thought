# phase 0, no rnn, gptmedium, interval 0
export INTERVAL=1
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export MODELSAVE="${MODEL////_}"
export SAVE=400kprelayernorm/phase0/interval${INTERVAL}/${MODEL}/r${R}_m${MODE}_e${EPOCHS}_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores_prelayernorm.py \
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
    --max_new_tokens 30 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=0
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export MODELSAVE="${MODEL////_}"
export SAVE=400kprelayernorm/phase0/interval${INTERVAL}/${MODEL}/r${R}_m${MODE}_e${EPOCHS}_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores_prelayernorm.py \
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
    --max_new_tokens 30 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=2
export MODEL=gpt2-medium
export F=diagonal
export EPOCHS=10
export LR=5e-5
export BSZ=32
export R=0
export MINUS=0
export MODE=bottom
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export MODELSAVE="${MODEL////_}"
export SAVE=400kprelayernorm/phase0/interval${INTERVAL}/${MODEL}/r${R}_m${MODE}_e${EPOCHS}_f${F}_minus${MINUS}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase0_fixq_math_fixed_interval_mlp_normgrad_nores_prelayernorm.py \
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
    --max_new_tokens 30 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
