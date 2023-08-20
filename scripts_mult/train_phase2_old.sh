
export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_mimic_fixq_feedp/checkpoint_6_5e-05
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mimic_fixq_feedp_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > models_mimic_fixq_feedp_phase2/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_mimic_fixq_feedp_gptsmall_top_gpt2_multmlp/checkpoint_1_5e-05
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_phase2_postfix/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_mimic_fixq_feedp_gptsmall_top_gpt2_multmlp/checkpoint_1_5e-05
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_phase2_postfix/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ}.rerun 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_mimic_fixq_feedp_gptsmall_top_gpt2_multmlp/checkpoint_1_5e-05
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mimic_fixq_feedp_phase2.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_phase2_postfix_saveq/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ}.rerun 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=models_mimic_fixq_feedp_phase2_postfix_saveq/checkpoint_1_5e-05
export QMODEL=gpt2
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_phase2_eval.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_phase2_postfix_saveq/log.train.text.modelgpt2.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ}.rerun.eval 2>&1&
