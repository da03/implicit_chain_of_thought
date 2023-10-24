export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=1
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=0.2
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=5
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=0.05
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_fixq
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --fix_q 1 \
    --mode top \
    --accumulate $A \
    --mixture_size $M \
    --additional_norm 0 \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=0.005
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=29
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=1
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=1
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --fix_p 1 \
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=0.2
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --fix_p 1 \
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=0.05
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_fixp
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --fix_p 1 \
    --softmax_p_temp $T \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=1
export PW=1
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_pw${PW}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --fix_p 0 \
    --p_mean_weight $PW \
    --softmax_p_temp $T \
    --interval $INTERVAL \
    --follow $F \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=1
export M=1
export E=19
export EPOCHS=10
export LR=5e-5
export FOLDER=sharps_nobrackets_300kaugmented_math_scaffolding_formula
export F=diagonal
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kprelayernorm/phase0/interval1/nornn/gptmedium/r0_mbottom_e10_fdiagonal_minus0/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300kpreliminary/nobrackets/use_attn_1rnn_predicttoken_prelayernorm_debugautoregressive/phase1/nolegacy/origdata/interval1/gptmedium/m1/feedp_useargmin_lr5e-5_w0_pw1_minus0_fdiagonal/checkpoint_${E}_5e-05 \
export BSZ=32
export A=1
export T=1
export PW=0.1
export SAVE=300k_phase2/nolegacy/origdata/gptmedium/m1/interval${INTERVAL}/nornn/feedp_useargmin/e${EPOCHS}_m${MODEL}_m${M}_w${W}_fixq_f${F}_unfix_e${E}_t${T}_pw${PW}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn_normgrad_predicttoken_prelayernorm_unfix_nornn.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --eval_path data/${FOLDER}/src1_train.txt.1000plussub1000 \
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
    --fix_p 0 \
    --p_mean_weight $PW \
    --softmax_p_temp $T \
    --interval $INTERVAL \
    --follow $F \
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
