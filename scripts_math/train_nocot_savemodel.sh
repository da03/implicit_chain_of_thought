for MODEL in gpt2 EleutherAI/gpt-neo-125m gpt2-medium xhyi/PT_GPTNEO350_ATG
do
    for D in scaffolding_none
    do
        echo "model: $MODEL D: $D"
        export EPOCHS=15
        export LR=5e-5
        #export MODEL=gpt2
        export BSZ=32
        #export D=scaffolding_formula
        export FOLDER=math_${D}
        export MODELSAVE="${MODEL////_}"
        export SAVE=math_${D}_nocot_${MODELSAVE}
        echo $SAVE
        mkdir $SAVE
        TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_nocot_savemodel_math.py \
            --train_path data/${FOLDER}/src1_train.txt \
            --val_path data/${FOLDER}/src1_valid.txt \
            --test_path data/${FOLDER}/src1_test.txt \
            --epochs $EPOCHS \
            --lr $LR \
            --model $MODEL \
            --batch_size $BSZ \
            --save_model $SAVE \
            --compile 0 \
            > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1
    done
done
