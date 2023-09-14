# interval 2 debug
export M=8
export FEED=p
export USE=argmin
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
export SAVE=autoregressive_sharps_augmented_model_${D}_phase1_inter_F${F}_gpt2_fixed_e${EPOCHS}_legacy_${CKPTEPOCH}_m${MODELSAVE}_try1_${CKPT}_repharvard_interval${INTERVAL}_100k_addmlp_nolegacy_rnn_mixture_m${M}_learnp_singlemlp_feed${FEED}_use${USE}
mkdir $SAVE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false stdbuf -oL -eL python train_phase1_fixed_math_interval_autoregressive_addmlp_rnn_mixture_learnp_feedp.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --qmodel $QMODEL \
    --mode top \
    --follow $F \
    --feed $FEED \
    --use $USE \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    --interval $INTERVAL \
    --mixture_size $M \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
