export INTERVAL=1
export MODEL=gpt2-medium
export LR=5e-5
export W=0
export MINUS=0
export PW=1
export M=1
export FEED=p
export USE=argmin
export EPOCHS=40
export D=scaffolding_formula
export BSZ=32
export F=diagonal
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval${INTERVAL}/pw${PW}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt\
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --last_id_minus $MINUS \
    --follow $F \
    --feed $FEED \
    --use $USE \
    --no_save 0 \
    --kl_mean_weight $W \
    --p_mean_weight $PW \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn 1 \
    --use_attn 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=0
export MODEL=gpt2-medium
export LR=5e-5
export W=0
export MINUS=0
export PW=1
export M=1
export FEED=p
export USE=argmin
export EPOCHS=40
export D=scaffolding_formula
export BSZ=32
export F=diagonal
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval${INTERVAL}/pw${PW}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt\
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --last_id_minus $MINUS \
    --follow $F \
    --feed $FEED \
    --use $USE \
    --no_save 0 \
    --kl_mean_weight $W \
    --p_mean_weight $PW \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn 1 \
    --use_attn 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=2
export MODEL=gpt2-medium
export LR=5e-5
export W=0
export MINUS=0
export PW=1
export M=1
export FEED=p
export USE=argmin
export EPOCHS=40
export D=scaffolding_formula
export BSZ=32
export F=diagonal
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval${INTERVAL}/pw${PW}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt\
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --last_id_minus $MINUS \
    --follow $F \
    --feed $FEED \
    --use $USE \
    --no_save 0 \
    --kl_mean_weight $W \
    --p_mean_weight $PW \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn 1 \
    --use_attn 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
