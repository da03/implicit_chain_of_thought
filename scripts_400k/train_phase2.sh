    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
export INTERVAL=1
export M=1
export E=17
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kprelayernorm/phase0/interval1/gpt2/r0_mbottom_e10_fdiagonal_minus0/checkpoint_3_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval1/pw1/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=32
export A=1
export T=0.05
export FIXP=0
export SAVE=400k_phase2/nolegacy/origdata/gptsmall/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp${FIXP}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --softmax_p 1 \
    --fix_p ${FIXP} \
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export INTERVAL=0
export M=1
export E=17
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kprelayernorm/phase0/interval0/gpt2/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval0/pw1/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=32
export A=1
export T=0.05
export FIXP=0
export SAVE=400k_phase2/nolegacy/origdata/gptsmall/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp${FIXP}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --softmax_p 1 \
    --fix_p ${FIXP} \
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
export M=1
export E=17
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kprelayernorm/phase0/interval2/gpt2/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval2/pw1/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=32
export A=1
export T=0.05
export FIXP=0
export SAVE=400k_phase2/nolegacy/origdata/gptsmall/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp${FIXP}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --softmax_p 1 \
    --fix_p ${FIXP} \
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
