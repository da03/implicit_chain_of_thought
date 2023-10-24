
# phase 1, no legacy, orig data, gptsmall, interval 0, m1, feedp, use argmin 
export INTERVAL=1
export MODEL=gpt2-medium
export LR=5e-5
export W=0
export MINUS=0
export PW=1
export M=1
export FEED=q
export USE=gt
export EPOCHS=60
export D=scaffolding_formula
export BSZ=32
export F=diagonal
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export SAVE=0rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
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
    --use_rnn 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
