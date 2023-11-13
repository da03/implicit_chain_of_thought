# GPT-4 Turbo V.S. GPT-4 Comparison

According to OpenAI, their new GPT-4 Turbo model is "3X cheaper for input tokens and 2X cheaper for output tokens compared to the original GPT-4 model" ([Schade, 2023](https://help.openai.com/en/articles/8555510-gpt-4-turbo)). But is GPT-4 Turbo as good as GPT-4? To answer this question, we compared GPT-4 Turbo to GPT-4 on three tasks requiring reasoning: 4-by-4 multiplication ([BIG-bench](https://github.com/google/BIG-bench)), 5-by-5 multiplication ([BIG-bench](https://github.com/google/BIG-bench)), and grade school math problems ([GSM8K](https://github.com/openai/grade-school-math)). Note that this set of experiments are adapted from the baselines of our [Implicit Chain of Thought Reasoning via Knowledge Distillation](https://arxiv.org/pdf/2311.01460.pdf) paper.

## Results

|                  |  **4X4** | **Mult**   |   |  **5X5** | **Mult**   |   |  **GSM** | **8K**     |
|------------------|---------:|------------|---|---------:|------------|---|---------:|------------|
|                  | Accuracy | Throughput |   | Accuracy | Throughput |   | Accuracy | Throughput |
| **No CoT**       |          |            |   |          |            |   |          |            |
| GPT-4            |   3.8%   |    1.04    |   |   0.1%   |    1.02    |   |   42.8%  |    1.05    |
| GPT-4 Turbo      |   6.1%   |    1.84    |   |   0.3%   |    1.71    |   |   43.3%  |    1.79    |
| **Explicit CoT** |          |            |   |          |            |   |          |            |
| GPT-4            |   77.8%  |    0.09    |   |   43.2%  |    0.07    |   |   90.9%  |    0.10    |
| GPT-4 Turbo      |   76.6%  |    0.31    |   |   38.3%  |    0.24    |   |   91.4%  |    0.31    |

Note that some results are slightly differently from our paper due to us rerunning all baseline evaluations (current evaluations run on Nov 12, 2023).

## Usage

Evaluation scripts of different settings (`no_cot` v.s. `explicit_cot`) and different tasks (`4_by_4_mult`, `5_by_5_mult`, and `gsm8k`) are organized into different folders. The only two required arguments that need to be provided are `--api_key` and `--model`, and there is an optional argument `--overwrite_cache` which makes a new query to OpenAI's API and overwrites the cache file even when an exact same query has been made and cached before.

* `--api_key`: specifies the OpenAI API key
* `--model`: specifies the OpenAI model to run, such as gpt-4 or gpt-4-1106-preview
* `--overwrite_cache` (optional): if specified, always make new queries to OpenAI's API and overwrites the local cache

For example, in order to evaluate GPT-4 Turbo's performance on GSM8K using explicit chain-of-thought reasoning, use the below command:

```
export APIKEY=your_openai_api_key
python explicit_cot/gsm8k/evaluate.py --api_key $APIKEY --model gpt-4-1106-preview
```

Note that in order to ensure the reproducibility of results, we have included all cache files in this repo. By default the cache files are used, in which case only the accuracy measurement is meaningful, whereas the throughput measurement is not.

## Acknowledgements

Our table above is inspired by [Chain-of-Thought Hub: Measuring LLMs' Reasoning Performance](https://github.com/FranxYao/chain-of-thought-hub), which contains a more comprehensive comparision of LLM's reasoning performance across a wider range of models and a wider range of tasks.
