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
