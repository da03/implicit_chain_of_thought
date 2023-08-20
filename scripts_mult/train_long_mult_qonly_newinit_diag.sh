export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mult_qonly_diag.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mult_qonly_diag_sig0.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_sig0_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_mult_qonly_diag_kl.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mult_qonly_diag_kl_noclip.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_klnoclip_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mult_qonly_diag_kl_disabledropout.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_nodropout_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mult_qonly_diag_kl_noclip_sgd.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_klnoclip_sgd_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_mult_qonly_diag_kl_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_fixq_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_mult_qonly_diag_kl_fixq_embed.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_fixq_embed_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mult_qonly_diag_kl_fixq_embed_trainq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_fixq_trainq_embed_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mult_qonly_diag_kl_fixq_embed_mlp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_fixq_mlp_embed_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mult_qonly_diag_kl_fixq_embed_mlp_trueemb.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > mult_kl_fixq_mlp_trueemb_embed_qonly_newinit_diag/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_mimic_fixq.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > models_mimic_fixq/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_mimic_fixq_feedp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > models_mimic_fixq_feedp/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export EPOCHS=15
export LR=5e-5
export FOLDER=long_mult
export MODEL=gpt2-medium
export QMODEL=/n/rush_lab/Lab/Users/yuntian/implicit/models_cot/checkpoint_1
export BSZ=32
CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python train_mimic_fixq_scratch.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    > models_mimic_fixq_scratch/log.train.text.model${MODEL}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&

