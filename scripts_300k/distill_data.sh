export I=0
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export EPOCHS=1
export LR=5e-5
export BSZ=32
export BEAM=1
export FOLDER=nobrackets_300kaugmented_math_scaffolding_formula_$I
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam${BEAM}_${I}
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python distill_cot_savemodel_math_folder.py \
    --folder data/${FOLDER} \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export I=1
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export EPOCHS=1
export LR=5e-5
export BSZ=32
export BEAM=1
export FOLDER=nobrackets_300kaugmented_math_scaffolding_formula_$I
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam${BEAM}_${I}
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1 stdbuf -oL -eL python distill_cot_savemodel_math_folder.py \
    --folder data/${FOLDER} \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export I=2
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export EPOCHS=1
export LR=5e-5
export BSZ=32
export BEAM=1
export FOLDER=nobrackets_300kaugmented_math_scaffolding_formula_$I
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam${BEAM}_${I}
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python distill_cot_savemodel_math_folder.py \
    --folder data/${FOLDER} \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
export I=3
export MODEL=/n/holyscratch01/rush_lab/Users/yuntian/implicit/300k_baselines/cot_nobrackets/gptmedium/checkpoint_6_5e-05_gpt2-medium
export EPOCHS=1
export LR=5e-5
export BSZ=32
export BEAM=1
export FOLDER=nobrackets_300kaugmented_math_scaffolding_formula_$I
export MODELSAVE="${MODEL////_}"
export SAVE=data/distilled_nobrackets_300kaugmented_math_scaffolding_formula_cot_gpt2-medium_repharvard_100k_beam${BEAM}_${I}
echo $SAVE
mkdir $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python distill_cot_savemodel_math_folder.py \
    --folder data/${FOLDER} \
    --epochs $EPOCHS \
    --lr $LR \
    --model $MODEL \
    --batch_size $BSZ \
    --save_data $SAVE \
    --compile 0 \
    --beam_size $BEAM \
    > ${SAVE}/log.train.text.model${MODELSAVE}.folder${FOLDER}.e${EPOCHS}.lr${LR}.${BSZ} 2>&1&
