# interval 2
export INTERVAL=2
export M=64
export W=0.001
export EPOCHS=10
export D=scaffolding_formula
export LR=5e-5
export QCKPTEPOCH=12
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/mixture_legacy/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e30_legacy_2_mgpt2_try1_6_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_w${W}_legacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_legacybest
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_legacy.py \
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
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/mixture_legacy/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e30_legacy_2_mgpt2_try1_6_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_w${W}_legacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_legacylast
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_legacy.py \
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
export QCKPTEPOCH=12
export FOLDER=sharps_augmented_math_${D}
export F=diagonal
export CKPTEPOCH=3
export CKPT=6
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_f${F}_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/mixture_legacy/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e30_legacy_2_mgpt2_try1_6_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_w${W}_legacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_legacybest_fixq
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_legacy_fixq.py \
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
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/mixture_legacy/autoregressive_sharps_augmented_model_scaffolding_formula_phase1_inter_Fdiagonal_gpt2_fixed_e30_legacy_2_mgpt2_try1_6_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feedp_useargmin_w${W}_legacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=autoregressive_sharps_augmented_${D}_phase2_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_bsz${BSZ}_nolegacy_e${EPOCHS}_m${MODELSAVE}_legacybest_interval${INTERVAL}_nolegacy_rnn_mixture_learnp_feedp_m${M}_w${W}_legacylast_fixq
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy_math_autoregressive_addmlp_nolegacy_rnn_mixture_learnp_feedp_legacy_fixq.py \
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
