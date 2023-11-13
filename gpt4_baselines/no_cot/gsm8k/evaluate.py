import hashlib
import numpy as np
import time
import sys, os, json, random, argparse, openai
import tiktoken
import tqdm
import sqlite3


def create_database(cache_file):
    conn = sqlite3.connect(cache_file)
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS main
                 (key TEXT PRIMARY KEY, prompt TEXT, completion TEXT)''')
    conn.commit()
    conn.close()


def insert_or_update(key, prompt, completion, cache_file):
    conn = sqlite3.connect(cache_file)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO main
                 (key, prompt, completion) VALUES (?, ?, ?)''', 
                 (key, prompt, completion))
    conn.commit()
    conn.close()

def retrieve(key, cache_file):
    conn = sqlite3.connect(cache_file)
    c = conn.cursor()

    c.execute("SELECT prompt, completion FROM main WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return (True, result)
    else:
        return (False, None)

def get_completion_with_cache(prompt, model, temperature, max_tokens, cache_file, overwrite_cache):
    #import pdb; pdb.set_trace()
    key = hashlib.sha256(json.dumps({'prompt': prompt, 'model': model, 'temperature': temperature}).encode('utf-8')).hexdigest()
    hit, result = retrieve(key, cache_file)
    if overwrite_cache:
        hit = False
    if not hit:
        completion = get_completion(prompt, model, temperature, max_tokens)
        insert_or_update(key, json.dumps(prompt), completion, cache_file)
    else:
        print ('hit')
        prompt, completion = result
    return completion, hit


#@retry(wait=wait_exponential(min=60, max=1200), stop=stop_after_attempt(12))
def chatcompletion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_completion(prompt, model, temperature, max_tokens):
    """
    Get a completion using the specified language model.

    Args:
        prompt (list): The prompt for generating the sentence.
        model (str): The name of the language model to use.
        temperature (float): Sampling temperature for the model.
        max_tokens (int): Maximum number of tokens in the generated sentence.

    Returns:
        str: The generated sentence.
    """
    generated_text = ""
    response = chatcompletion_with_backoff(
        model=model,
        messages=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    generated_text = response.choices[0].message.content.strip()
    return generated_text

def construct_prompt(num_shot, examples):
    instruction = 'Answer the final question following the exact format of the given examples. Do not break the problem down, directly produce the answer.\n\nExample problems:\n\n'
    example_demonstrations = random.sample(examples, num_shot)
    prompt = instruction
    for i, line in enumerate(example_demonstrations):
        s = f"Q: {line['input']}\n"
        s += f"A: #### {line['output']}\n"
        prompt += s
    prompt += '\n'
    prompt += f"Question to answer:\n\nQ:"
    context = [{"role": 'user', "content": prompt}]
    return context

def read_examples(filename):
    lines = []
    with open(filename) as fin:
        for line in fin:
            input, cot_and_output = line.strip().split('||')
            cot, output = cot_and_output.split(' #### ')
            input = input.strip()
            cot = cot.strip()
            output = output.strip()
            lines.append({'input': input, 'output': output, 'cot': cot})
    return lines

def main(model, temperature, max_tokens, seed, num_shot, train_file, test_file, cache_file, overwrite_cache):
    create_database(cache_file)
    random.seed(seed)
    train_examples = read_examples(train_file)
    test_examples = read_examples(test_file)
    
    prompts = []
    for _ in test_examples:
        prompt = construct_prompt(num_shot, train_examples)
        prompts.append(prompt)

    i = 0
    correct = 0
    total = 0
    total_time = 0
    not_hit = 0
    for example in test_examples:
        prompt = prompts[i]
        i += 1
        prompt[0]['content'] += example['input']
        start_time = time.time()
        completion, hit = get_completion_with_cache(prompt, model, temperature, max_tokens, cache_file, overwrite_cache)
        answer = completion.split('####')[-1].strip()
        if not hit:
            not_hit += 1
            end_time = time.time()
            total_time += end_time - start_time
            print (f'throughput: {not_hit / total_time}, total time: {total_time}, total number of examples: {not_hit}')
        if answer == example['output']:
            correct += 1
        total += 1
        print (f'accuracy: {correct / total}, correct: {correct}, total: {total}')
        sys.stdout.flush()
    if not_hit > 0:
        print (f'final throughput: {not_hit / total_time}, total time: {total_time}, total number of examples: {not_hit}')
    print (f'final accuracy: {correct / total}, correct: {correct}, total: {total}')


def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Augment")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default='gpt-4-1106-preview', help="model to evaluate")
    parser.add_argument("--train_file", type=str, default="../data/gsm8k/train_no_aug.txt")
    parser.add_argument("--test_file", type=str, default="../data/gsm8k/test.txt")
    parser.add_argument("--num_shot", type=int, default=5)
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument("--cache_file", type=str, default="no_cot/gsm8k/cache.db")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=600, help="Maximum number of tokens in the generated sentence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.set_defaults(overwrite_cache=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    openai.api_key = args.api_key

    main(model=args.model, \
         temperature=args.temperature, \
         max_tokens=args.max_tokens, \
         seed=args.seed, \
         train_file=args.train_file, \
         test_file=args.test_file, \
         num_shot=args.num_shot, \
         cache_file=args.cache_file, \
         overwrite_cache=args.overwrite_cache)
