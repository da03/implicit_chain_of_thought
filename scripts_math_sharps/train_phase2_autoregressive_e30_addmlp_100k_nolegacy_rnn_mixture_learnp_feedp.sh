# interval 2
export INTERVAL=2
export M=64
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=29
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/mixture/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e30_legacy_2_mgpt2_try1_6_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_w${W}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
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
export M=64
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=29
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/mixture/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e30_legacy_2_mgpt2_try1_6_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_w${W}/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
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
export QCKPTEPOCH=59
export FOLDER=sharps_augmented_math_${D}
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fbottom_row_opt_6_repharvard_interval2_100k_fbottom_row/checkpoint_3_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/fixnorm_mixture/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2_try1_6_repharvard_interval2_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_lr5e-5_w${W}_pw0.01_minus0_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05 \
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
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
export QCKPTEPOCH=20
export FOLDER=sharps_augmented_math_${D}
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fbottom_row_opt_6_repharvard_interval2_100k_fbottom_row/checkpoint_3_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/fixnorm_mixture/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2_try1_6_repharvard_interval2_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_lr5e-5_w${W}_pw0.01_minus0_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05 \
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
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
export QCKPTEPOCH=20
export FOLDER=sharps_augmented_math_${D}
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fbottom_row_opt_6_repharvard_interval2_100k_fbottom_row/checkpoint_3_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/fixnorm_mixture/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2_try1_6_repharvard_interval2_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_lr5e-5_w${W}_pw0.01_minus0_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05 \
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
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
export QCKPTEPOCH=59
export FOLDER=sharps_augmented_math_${D}
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fbottom_row_opt_6_repharvard_interval2_100k_fbottom_row_minus3/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/minus_phase1/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2_try1_6_repharvard_interval2_100k_addmlp_nolegacy_rnn_mixture_m8_learnp_singlemlp_feedp_useargmin_lr5e-5_w0.001_pw0.01_minus3_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
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
export QCKPTEPOCH=59
export FOLDER=sharps_distilled_augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k_beam1
export F=bottom_row
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2-medium_r0_mbottom_e10_fbottom_row_opt_5_repharvard_interval1_100k_fbottom_row_minus3_interval1/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/gpt2medium_minus_phase1_distill/autoregressive_sharps_scaffolding_formula_phase1_inter_Fbottom_row_gpt2_fixed_e60_legacy_2_mgpt2-medium_try1_5_interval1_100k_addmlp_nolegacy_rnn_mixture_m8_learnp_singlemlp_feedq_usegt_lr5e-5_w0.001_pw0.01_minus3_fbottom_row/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
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
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_e9
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
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
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_e${QCKPTEPOCH}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
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
export SAVE=gpt2medium_minus_phase2/autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_lastckpt_e${QCKPTEPOCH}
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp.py \
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
export INTERVAL=0
export M=1
export W=0
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=16
export FOLDER=sharps_200kaugmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200k_minus_phase0/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_6_repharvard_interval0_100k_fdiagonal_minus0_normgrad_noresidual/checkpoint_2_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200k_minus_phase1/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e60_legacy_2_mgpt2_try1_6_repharvard_interval0_100k_addmlp_nolegacy_rnn_mixture_m1_learnp_singlemlp_feedp_useargmin_lr5e-5_w0_pw0_minus0_fdiagonal/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=200k_minus_phase2/e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_m${M}_w${W}_fixq_f${F}_bestval_argmin_nofixq_bestvallast
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_nornn.py \
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
    > ${SAVE}/log.train.text.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
