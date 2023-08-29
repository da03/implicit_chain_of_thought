export EPOCHS=15
export LR=5e-5
export D=4
export FOLDER=long_mult_${D}_inter
export MODEL=gpt2-medium
export CKPTEPOCH=1
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_postphase0_gptmedium_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_gptmedium
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=4
export FOLDER=long_mult_${D}_inter
export MODEL=gpt2-medium
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_postphase0_gptmedium_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_gptmedium
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=4
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_gptmedium_nopretrain/checkpoint_1_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_postphase0_gptmedium_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_gptmedium_initmodel
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=2
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_gptlarge_nopretrain/checkpoint_1_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_postphase0_gptlarge_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_gptlarge_initmodel
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=5
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_gptmedium_nopretrain/checkpoint_1_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_postphase0_gptmedium_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_gptmedium_initmodel
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=6
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_gptmedium_nopretrain/checkpoint_1_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_postphase0_gptmedium_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_gptmedium_initmodel
mkdir $SAVE
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=14
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_fixq/checkpoint_6_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_fixq_initmodel
mkdir $SAVE
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=14
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_fixq/checkpoint_9_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_fixq_initmodel_pckpt9
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=4
export CKPTEPOCH=14
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_4_inter_fixq/checkpoint_10_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter/checkpoint_${CKPTEPOCH}_5e-05
export BSZ=32
export SAVE=model_phase2_${D}_inter_${CKPTEPOCH}_postphase0_fixq_initmodel_pckpt10
mkdir $SAVE
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=4
export QCKPTEPOCH=3
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2xl_pgpt2xl_r0_mbottom_bf32/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2xl_bf32/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2xl${CKPTEPOCH}_qgpt2xl${QCKPTEPOCH}_postphase0_fixq_initmodel_opt
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=4
export QCKPTEPOCH=3
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2xl_pgpt2xl_r0_mbottom_bf32/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2xl_bf32/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=64
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2xl${CKPTEPOCH}_qgpt2xl${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz64
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=4
export QCKPTEPOCH=3
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2xl_pgpt2xl_r0_mbottom_bf32/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2xl_bf32/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=128
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2xl${CKPTEPOCH}_qgpt2xl${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=16
export QCKPTEPOCH=14
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=16
export QCKPTEPOCH=12
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_16/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=16
export QCKPTEPOCH=23
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_16/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=24
export QCKPTEPOCH=29
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_4_cont12_cont16/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy_cont12_cont16
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=4
export QCKPTEPOCH=29
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_4_cont12_cont16/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy_cont12_cont16_fixed
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=24
export QCKPTEPOCH=42
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_5_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_24_cont12_cont16_latestaug25/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy_cont12_cont16_fixed_latestaug25
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=24
export QCKPTEPOCH=59
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_5_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_24_cont12_cont16_latestaug25/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy_cont12_cont16_fixed_latestaug26
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export D=6
export CKPTEPOCH=44
export QCKPTEPOCH=22
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_6_inter_fixq_qgpt2_pgpt2_r0_mbottom_e120_fdiagonal_opt/checkpoint_${CKPTEPOCH}_5e-05
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_5_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_24_cont12_cont16_latestaug25/checkpoint_${QCKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_6_inter_Fdiagonal_gpt2_fixed_e60_legacy_44/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&



export EPOCHS=15
export LR=5e-5
export D=5
export CKPTEPOCH=24
export QCKPTEPOCH=14
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=60
export LR=5e-5
export D=6
export CKPTEPOCH=44
export QCKPTEPOCH=22
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_6_inter_fixq_qgpt2_pgpt2_r0_mbottom_e120_fdiagonal_opt/checkpoint_${CKPTEPOCH}_5e-05
#export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_5_inter_Fdiagonal_orig_gpt2_fixed_e60_legacy_24_cont12_cont16_latestaug25/checkpoint_${QCKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_6_inter_Fdiagonal_gpt2_fixed_e60_legacy_44/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_fixed_legacy_e${EPOCHS}_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp_fixed_legacy.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&



export EPOCHS=40
export LR=5e-5
export D=5
export CKPTEPOCH=24
export QCKPTEPOCH=59
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_nolegacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=40
export LR=5e-5
export D=5
export CKPTEPOCH=47
export QCKPTEPOCH=59
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_5_inter_fixq_qgpt2_pgpt2_r0_mbottom_e120_fdiagonal_orig_opt_try1/checkpoint_${CKPTEPOCH}_5e-05
#/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_nolegacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_nocontmodel
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=40
export LR=5e-5
export D=5
export CKPTEPOCH=59
export QCKPTEPOCH=59
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_5_inter_fixq_qgpt2_pgpt2_r0_mbottom_e120_fdiagonal_orig_opt_try1/checkpoint_${CKPTEPOCH}_5e-05
#/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_orig_gpt2_fixed_e60_nolegacy/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2${CKPTEPOCH}_qgpt2${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_nocontmodel_laterepoch
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=40
export LR=5e-5
export D=5
export CKPTEPOCH=19
export QCKPTEPOCH=43
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_5_inter_fixq_qgpt2medium_pgpt2-medium_r0_mbottom_e120_fdiagonal_opt_try1/checkpoint_${CKPTEPOCH}_5e-05
#/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_gpt2medium_fixed_e60_nolegacy_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2medium${CKPTEPOCH}_qgpt2medium${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-medium.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=40
export LR=5e-5
export D=5
export CKPTEPOCH=19
export QCKPTEPOCH=23
export FOLDER=long_mult_${D}_inter
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_5_inter_fixq_qgpt2large_pgpt2-large_r0_mbottom_e120_fdiagonal_opt_try1/checkpoint_${CKPTEPOCH}_5e-05
#/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase0_${D}_inter_fixq_qgpt2_pgpt2_r0_mbottom_e60_cont12_cont16/checkpoint_${CKPTEPOCH}_5e-05
export QMODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/model_phase1_${D}_inter_Fdiagonal_gpt2large_fixed_e60_nolegacy_try1/checkpoint_${QCKPTEPOCH}_5e-05
export BSZ=32
export A=1
export SAVE=model_phase2_${D}_inter_pgpt2large${CKPTEPOCH}_qgpt2large${QCKPTEPOCH}_postphase0_fixq_initmodel_opt_bsz${BSZ}_nolegacy_e${EPOCHS}_try1
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_phase2_patchmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --mode top \
    --accumulate $A \
    > ${SAVE}/log.train.text.modelgpt2-large.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

