export INTERVAL=0
export M=1
export E=4
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_ans_long_mult_4_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/ans_long_mult_4_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_1_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/ans_long_mult_4_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=32
export A=1
export T=1
export SAVE=ans_long_mult_4_inter_phase2/nomixture/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
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
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=0
export M=1
export E=4
export EPOCHS=40
export LR=5e-5
export FOLDER=sharps_ans_long_mult_5_inter
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/ans_long_mult_5_prelayernorm/phase0/interval0/gpt2-medium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_4_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/ans_long_mult_5_prelayernorm/use_attn_1rnn_predicttoken_prelayernorm/phase1/nomixture/nolegacy/interval0/pw0/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${E}_5e-05
export BSZ=32
export A=1
export T=1
export SAVE=ans_long_mult_5_inter_phase2/nomixture/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test_bigbench.txt \
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
    --no_mixture 1 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
