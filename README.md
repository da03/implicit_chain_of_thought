# Implicit Chain of Thought Reasoning via Knowledge Distillation

Here we provide code to reproduce our results.

## Prerequisites

* [PyTorch](https://pytorch.org/get-started/locally/)
* [transformers](https://github.com/huggingface/transformers) (`pip install transformers`)

## Datasets & Pretrained Models & Logs

All dataset files and log files during inference are included in this repo, with the exception of large training files maintained using Git LFS. Model checkpoints are stored on Google Drive. The folder containing all checkpoints can be found at [this link](https://drive.google.com/drive/folders/1Sclr5bmLZIUcktCaFAeWRTevRGLUwlC_?usp=drive_link).

* 4 X 4 Mult - GPT-2: [data](data/4_by_4_mult/) [model](https://drive.google.com/drive/folders/1Zp-PFwiHkwq0wuFScjN5R8jDdXdnQYQ_?usp=sharing) [log](logs/4_by_4_mult/gpt2/log.generate)
* 4 X 4 Mult - GPT-2 Medium: [data](data/4_by_4_mult/) [model](https://drive.google.com/drive/folders/1B0e67ifTSTTuUg0Sh-of5135Rh4KQ-2v?usp=sharing) [log](logs/4_by_4_mult/gpt2-medium/log.generate)
* 5 X 5 Mult - GPT-2: [data](data/5_by_5_mult/) [model](https://drive.google.com/drive/folders/1lHa2Xey8jJ3__RsYRhcOFHU7Xfqp7XTG?usp=sharing) [log](logs/5_by_5_mult/gpt2/log.generate)
* 5 X 5 Mult - GPT-2 Medium: [data](data/5_by_5_mult/) [model](https://drive.google.com/drive/folders/18dRIynq0j5EBOnKTpOPaLJWCoMBXZYTi?usp=sharing) [log](logs/5_by_5_mult/gpt2-medium/log.generate)
* GSM8K - GPT-2: [data](data/5_by_5_mult/) [model](https://drive.google.com/drive/folders/1aFBBcUr_vHtaDqgpU5A1ErEvrJyX-cEO?usp=sharing) [log](logs/gsm8k/gpt2/log.generate)
* GSM8K - GPT-2 Medium: [data](data/5_by_5_mult/) [model](https://drive.google.com/drive/folders/1zFXfwq5jDjgKpbUVafY5KC0LmJpYXjQK?usp=sharing) [log](logs/gsm8k/gpt2-medium/log.generate)

## Usage

We use 5 X 5 Mult as an example.


### Training

#### 0. Teacher Training

#### 1. Mind-Reading the Teacher

#### 2. Thought Emulation

#### 3. Couple and Optimize


### Generation & Evaluation

Here we use a pretrained model as an example. Download the folder `models/5_by_5_mult/gpt2-medium`, then the following command will run inference and evaluate both accuracy and throughput, logged in file `generation_logs/5_by_5_mult/log.generate`.

```
export FOLDER=data/5_by_5_mult
export STUDENT=models/5_by_5_mult/gpt2-medium/student
export EMULATOR=models/5_by_5_mult/gpt2-medium/emulator
export BSZ=1
export SAVE=generation_logs/5_by_5_mult
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
