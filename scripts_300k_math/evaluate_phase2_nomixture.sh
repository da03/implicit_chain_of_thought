export INTERVAL=0
export M=1
export E=14
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_long_mult_4_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e10_fdiagonal_minus0_cont/checkpoint_5_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=1
export A=1
export T=1
export SAVE=eval_long_mult_4_inter_phase2/nomixture/origdata/gptmedium/m1/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_inter_phase2/nomixture/origdata/gptmedium/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e10_fdiagonal_minus0_cont/checkpoint_5_5e-05_m1_w_fixq_fdiagonal_unfix_e14_t1/checkpoint_0_5e-05 \
    --batch_size $BSZ \
    --qmodel /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_inter_phase2/nomixture/origdata/gptmedium/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e10_fdiagonal_minus0_cont/checkpoint_5_5e-05_m1_w_fixq_fdiagonal_unfix_e14_t1/checkpoint_0_5e-05/q \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=0
export M=1
export E=14
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_long_mult_4_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_3_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=1
export A=1
export T=1
export SAVE=eval_long_mult_4_inter_phase2/nomixture/origdata/gptsmall/m1/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_inter_phase2/nomixture/origdata/gptsmall/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_3_5e-05_m1_w_fixq_fdiagonal_unfix_e6_t1/checkpoint_6_5e-05 \
    --batch_size $BSZ \
    --qmodel /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_inter_phase2/nomixture/origdata/gptsmall/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_4_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_3_5e-05_m1_w_fixq_fdiagonal_unfix_e6_t1/checkpoint_6_5e-05/q \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=0
export M=1
export E=13
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_long_mult_5_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e40_fdiagonal_minus0_rerun/checkpoint_26_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=1
export A=1
export T=1
export SAVE=eval_long_mult_5_inter_phase2/nomixture/origdata/gptmedium/m1/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python evaluate_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_inter_phase2/nomixture/origdata/gptmedium/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e40_fdiagonal_minus0_rerun/checkpoint_26_5e-05_m1_w0_fixq_fdiagonal_unfix_e13_t1_lr1e5/checkpoint_3_1e-05 \
    --batch_size $BSZ \
    --qmodel /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_inter_phase2/nomixture/origdata/gptmedium/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e40_fdiagonal_minus0_rerun/checkpoint_26_5e-05_m1_w0_fixq_fdiagonal_unfix_e13_t1_lr1e5/checkpoint_3_1e-05/q \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=0
export M=1
export E=23
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_long_mult_5_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_38_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=1
export A=1
export T=1
export SAVE=eval_long_mult_5_inter_phase2/nomixture/origdata/gptmedium/m1/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python evaluate_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_inter_phase2/nomixture/origdata/gptsmall/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_38_5e-05_m1_w_fixq_fdiagonal_unfix_e23_t1_lr1e-5/checkpoint_2_1e-05 \
    --batch_size $BSZ \
    --qmodel /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_inter_phase2/nomixture/origdata/gptsmall/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_38_5e-05_m1_w_fixq_fdiagonal_unfix_e23_t1_lr1e-5/checkpoint_2_1e-05/q \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=0
export M=1
export E=23
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_long_mult_5_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_38_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptsmall/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=1
export A=1
export T=1
export SAVE=eval_long_mult_5_inter_phase2/nomixture/origdata/gptmedium/m1/interval${INTERVAL}/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_2
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python evaluate_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_inter_phase2/nomixture/origdata/gptsmall/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_38_5e-05_m1_w_fixq_fdiagonal_unfix_e23_t1_lr1e-5/checkpoint_5_1e-05 \
    --batch_size $BSZ \
    --qmodel /n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_inter_phase2/nomixture/origdata/gptsmall/m1/interval0/e40_m/n/holyscratch01/rush_lab/Users/yuntian/implicit/long_mult_5_prelayernorm/phase0/interval0/gpt2/r0_mbottom_e40_fdiagonal_minus0/checkpoint_38_5e-05_m1_w_fixq_fdiagonal_unfix_e23_t1_lr1e-5/checkpoint_5_1e-05/q \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
