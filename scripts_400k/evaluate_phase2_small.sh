export INTERVAL=2
export M=1
export E=17
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_400kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kprelayernorm/phase0/interval2/gpt2/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm/phase1/nolegacy/origdata/interval2/pw1/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=1
export A=1
export T=0.05
export FIXP=1
export SAVE=evaluate_400k_phase2/nolegacy/origdata/gptsmall/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp${FIXP}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python evaluate_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_phase2/nolegacy/origdata/gptsmall/interval2/e10_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kprelayernorm/phase0/interval2/gpt2/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05_m1_w_fixq_fdiagonal_unfix_e17_t0.05_fixp1/checkpoint_5_5e-05 \
    --qmodel /n/holyscratch01/rush_lab/Users/yuntian/implicit/400k_phase2/nolegacy/origdata/gptsmall/interval2/e10_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/400kprelayernorm/phase0/interval2/gpt2/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05_m1_w_fixq_fdiagonal_unfix_e17_t0.05_fixp1/checkpoint_5_5e-05/q \
    --batch_size $BSZ \
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
