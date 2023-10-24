export INTERVAL=1
export LR=5e-5
export W=0
export MINUS=0
export PW=1
export M=1
export FEED=p
export USE=argmin
export EPOCHS=60
export D=scaffolding_formula
export MODEL=gpt2
export BSZ=32
export F=diagonal
export FOLDER=sharps_200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval${INTERVAL}/gptsmall/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_29_5e-05 \
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
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
