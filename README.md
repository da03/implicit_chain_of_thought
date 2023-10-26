# Implicit Chain of Thought Reasoning via Knowledge Distillation

Here we provide code to reproduce our results.

## Prerequisites

* [PyTorch](https://pytorch.org/get-started/locally/)
* [transformers](https://github.com/huggingface/transformers)

## Datasets & Pretrained Models & Logs

* 4 X 4 Mult: [data](data/4_by_4_mult/) [model]() [log](logs/4_by_4_mult/log.generate)
* 5 X 5 Mult: [data](data/5_by_5_mult/) [model]() [log](logs/5_by_5_mult/log.generate)

## Usage

We use 5 X 5 Mult as an example.

### Data Preprocessing


### Training



### Generation & Evaluation

```
export FOLDER=data/5_by_5_mult
export STUDENT=models/5_by_5_mult/gpt2-medium/student
export EMULATOR=models/5_by_5_mult/gpt2-medium/emulator
export BSZ=1
export SAVE=logs/5_by_5_mult
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/generate.py \
    --batch_size $BSZ \
    --test_path ${FOLDER}/test_bigbench.txt \
    --student_path $STUDENT \
    --emulator_path $EMULATOR \
    > ${SAVE}/log.generate 2>&1&
```

## Citation

```
@inproceedings{
    anonymous2023implicit,
    title={Implicit Chain of Thought Reasoning via Knowledge Distillation},
    author={Anonymous},
    booktitle={Submitted to The Twelfth International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=9cumTvvlHG},
    note={under review}
}
```
