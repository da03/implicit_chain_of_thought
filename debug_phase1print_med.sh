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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/distilldata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/0rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_37_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

for E in 9 19 29 39
do
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/0rnn/nolegacy/distilldata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/0rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done


for E in 9 19 29 39
do
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=0
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/rmlayernorm/nolegacy/realdistilldata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/0rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/distilldata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done



# 1nn orig data
for E in 9 19 29 39
do
    export E=9
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/rmlayernorm/nolegacy/realorigdata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/1rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
for E in 9 19 29 39
do
    export E=9
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/realrmlayernorm/nolegacy/realorigdata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/rmlayernorm_1rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
for E in 9 19 29 39
do
    export E=9
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/realrmlayernorm/nolegacy/realdistilldata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/rmlayernorm_1rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/distilldata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
for E in 9 19 29 39
do
    export E=9
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/useattn/nolegacy/realorigdata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/use_attn_1rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 0 \
    --use_attn 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
for E in 9 19 29 39
do
    export E=9
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
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_200kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/useattn/nolegacy/realdistilldata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/use_attn_1rnn_predicttoken_200kprelayernorm_debugautoregressive/phase1/nolegacy/distilldata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 0 \
    --use_attn 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
for E in 29 38
do
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
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_300kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/useattn0/nolegacy/realorigdata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 0 \
    --use_attn 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
for E in 29 38
do
#    export E=8
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
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
#export FOLDER=sharps_distilled_200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam1
export MODELSAVE="${MODEL////_}"
export ONERNN=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export SAVE=printeqns_predicttoken_300kprelayernorm_debugautoregressive/phase1/gptmedium/${ONERNN}rnn/useattn1/nolegacy/realorigdata/interval${INTERVAL}/gptmedium/m${M}/feed${FEED}_use${USE}_lr${LR}_w${W}_pw${PW}_minus${MINUS}_f${F}_e${E}
mkdir -p $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp_fixnorm_prelayernorm_predicttoken.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --load_model 1 \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
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
    --eval_only 1 \
    --interval $INTERVAL \
    --mixture_size $M \
    --use_rnn ${ONERNN} \
    --no_layernorm 0 \
    --use_attn 1 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
done
