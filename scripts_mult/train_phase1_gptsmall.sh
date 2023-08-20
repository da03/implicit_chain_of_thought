export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_gptsmall_top_gpt2-medium/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > models_mimic_fixq_feedp_gptsmall_interleave_gpt2-medium/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_gptsmall_top_gpt2-xl/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > models_mimic_fixq_feedp_gptsmall_interleave_gpt2-xl/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_gptsmall_top_gpt2-medium_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > models_mimic_fixq_feedp_gptsmall_interleave_gpt2-medium_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > models_mimic_fixq_feedp_gptsmall_interleave_gpt2-xl_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-xl
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_gptsmall_top_gpt2-xl_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode interleave \
    > models_mimic_fixq_feedp_gptmedium_interleave_gpt2-medium_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1_5e-05_gpt2/
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_gptsmall_top_gpt2_multmlp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2
export QMODEL=gpt2
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp_gptsmall_multmlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    > models_mimic_fixq_feedp_gptsmall_top_gpt2_multmlp_gpt2q/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
