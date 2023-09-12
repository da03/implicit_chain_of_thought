# interval 0
export EPOCHS=30
export INTERVAL=0
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=3
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
# interval 1
export EPOCHS=30
export INTERVAL=1
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
# interval 2
export EPOCHS=30
export INTERVAL=2
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
# interval 3
export EPOCHS=30
export INTERVAL=3
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
# interval 4
export EPOCHS=30
export INTERVAL=4
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=3
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


# interval 2 debug
export EPOCHS=30
export INTERVAL=2
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k_addmlp
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive_addmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
# interval 0
export EPOCHS=30
export INTERVAL=0
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=3
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k_addmlp
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive_addmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


# interval 3
export EPOCHS=30
export INTERVAL=3
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=2
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k_addmlp
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive_addmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
# interval 4
export EPOCHS=30
export INTERVAL=4
export D=scaffolding_formula
export MODEL=gpt2
export LR=5e-5
export BSZ=32
export F=diagonal
export FOLDER=sharps_augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export CKPTEPOCH=3
export CKPT=6
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_${D}_cot_${MODEL}_repharvard_100k/checkpoint_${CKPT}_5e-05_gpt2
export AMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/sharps_augmented_model_scaffolding_formula_phase0_inter_fixq_qgpt2_pgpt2_r0_mbottom_e10_fdiagonal_opt_${CKPT}_repharvard_interval${INTERVAL}_100k/checkpoint_${CKPTEPOCH}_5e-05
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k_addmlp
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_legacy_math_interval_autoregressive_addmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --amodel $AMODEL \
    --mode top \
    --follow $F \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
