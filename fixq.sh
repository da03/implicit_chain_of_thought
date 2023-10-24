
# interval 2
export INTERVAL=2
export M=8
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=59
export FOLDER=sharps_distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2-medium_r0_mbottom_e10_fbottom_row_opt_5_repharvard_interval1_100k_fbottom_row_minus3_interval1/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase1_distill/autoregressive_sharps_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2-medium_try1_5_interval1_100k_addmlp_nolegacy_rnn_mixture_m8_learnp_singlemlp_feedq_usegt_lr5e-5_w0.001_pw0.01_minus3_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_fixq
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/sharps_augmented_math_scaffolding_formula/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt.train \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


# interval 2
export INTERVAL=2
export M=8
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=9
export FOLDER=sharps_distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2-medium_r0_mbottom_e10_fbottom_row_opt_5_repharvard_interval1_100k_fbottom_row_minus3_interval1/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase1_distill/autoregressive_sharps_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2-medium_try1_5_interval1_100k_addmlp_nolegacy_rnn_mixture_m8_learnp_singlemlp_feedq_usegt_lr5e-5_w0.001_pw0.01_minus3_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_e9_fixq
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/sharps_augmented_math_scaffolding_formula/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt.train \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
export M=8
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=19
export FOLDER=sharps_distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2-medium_r0_mbottom_e10_fbottom_row_opt_5_repharvard_interval1_100k_fbottom_row_minus3_interval1/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase1_distill/autoregressive_sharps_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2-medium_try1_5_interval1_100k_addmlp_nolegacy_rnn_mixture_m8_learnp_singlemlp_feedq_usegt_lr5e-5_w0.001_pw0.01_minus3_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_e${QCKPTEPOCH}_fixq
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/sharps_augmented_math_scaffolding_formula/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt.train \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export INTERVAL=2
export M=8
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=29
export FOLDER=sharps_distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2-medium_r0_mbottom_e10_fbottom_row_opt_5_repharvard_interval1_100k_fbottom_row_minus3_interval1/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase1_distill/autoregressive_sharps_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2-medium_try1_5_interval1_100k_addmlp_nolegacy_rnn_mixture_m8_learnp_singlemlp_feedq_usegt_lr5e-5_w0.001_pw0.01_minus3_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_e${QCKPTEPOCH}_fixq
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/sharps_augmented_math_scaffolding_formula/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt.train \
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
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
