export MODEL=gpt2
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=15
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
#export D=scaffolding_formula
export FOLDER=augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export SAVE=augmented_math_${D}_cot_${MODELSAVE}_repharvard_100k
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_6_5e-05_gpt2
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=1
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
export BEAM=1
#export D=scaffolding_formula
export FOLDER=augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_augmented_math_${D}_cot_gpt2_repharvard_100k_beam$BEAM
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python distill_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export MODEL=gpt2-large
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=15
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
#export D=scaffolding_formula
export FOLDER=augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export SAVE=augmented_math_${D}_cot_${MODELSAVE}_repharvard_100k
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/augmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_4_5e-05_gpt2-medium
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=1
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
export BEAM=1
#export D=scaffolding_formula
export FOLDER=augmented_math_${D}
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_augmented_math_${D}_cot_gpt2-medium_repharvard_100k_beam$BEAM
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python distill_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_test.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export MODEL=gpt2
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=15
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
#export D=scaffolding_formula
export FOLDER=200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=200kaugmented_math_${D}_cot_${MODELSAVE}_repharvard_100k
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export MODEL=gpt2-medium
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=15
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
#export D=scaffolding_formula
export FOLDER=200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=200kaugmented_math_${D}_cot_${MODELSAVE}_repharvard_100k
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python train_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_model /n/holyscratch01/rush_lab/Users/yuntian/implicit/$SAVE \
    --compile 0 \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2_repharvard_100k/checkpoint_14_5e-05_gpt2
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=1
export LR=5e-5
#export MODEL=gpt2
export BSZ=64
export BEAM=1
#export D=scaffolding_formula
export FOLDER=200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_200kaugmented_math_${D}_cot_gpt2_repharvard_100k_beam$BEAM
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python distill_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&


export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/200kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k/checkpoint_5_5e-05_gpt2-medium
export D=scaffolding_formula
echo "model: $MODEL D: $D"
export EPOCHS=1
export LR=5e-5
#export MODEL=gpt2
export BSZ=32
export BEAM=1
#export D=scaffolding_formula
export FOLDER=200kaugmented_math_scaffolding_formula
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_200kaugmented_math_${D}_cot_gpt2-medium_repharvard_100k_beam$BEAM
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python distill_cot_savemodel_math.py \
    --train_path data/${FOLDER}/src1_train.txt.1000plus \
    --val_path data/${FOLDER}/src1_valid.txt \
    --test_path data/${FOLDER}/src1_train.txt.1000 \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
