import re
import shutil
import hashlib
import numpy as np
import time
import sys, os, json, random, argparse, openai
import tiktoken
import tqdm
from multiprocessing import Pool

from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
)


import sqlite3


def parallel_function(args):
    return get_completion_with_cache(args[0], args[1], args[2], args[3])

def create_database():
    conn = sqlite3.connect('augmented.db')
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS augmented
                 (key TEXT PRIMARY KEY, prompt TEXT, completion TEXT)''')
    conn.commit()
    conn.close()

create_database()

def insert_or_update(key, prompt, completion):
    conn = sqlite3.connect('augmented.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO augmented
                 (key, prompt, completion) VALUES (?, ?, ?)''', 
                 (key, prompt, completion))
    conn.commit()
    conn.close()

def retrieve(key):
    conn = sqlite3.connect('augmented.db')
    c = conn.cursor()

    c.execute("SELECT prompt, completion FROM augmented WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    if result:
        return (True, result)
    else:
        return (False, None)

def get_completion_with_cache(prompt, model, temperature, max_tokens):
    #import pdb; pdb.set_trace()
    key = hashlib.sha256(json.dumps({'prompt': prompt, 'model': model, 'temperature': temperature}).encode('utf-8')).hexdigest()
    hit, result = retrieve(key)
    if not hit:
        return None
        completion = get_completion(prompt, model, temperature, max_tokens)
        insert_or_update(key, json.dumps(prompt), completion)
    else:
        print ('hit')
        prompt, completion = result
    return completion

@retry(wait=wait_exponential(min=1, max=1200), stop=stop_after_attempt(12))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@retry(wait=wait_exponential(min=1, max=1200), stop=stop_after_attempt(12))
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
    if 'gpt-3.5' in model or 'gpt-4' in model:
        #while len(generated_text) == 0:
        response = chatcompletion_with_backoff(
            model=model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        generated_text = response.choices[0].message.content.strip()
    else:
        #while len(generated_text) == 0:
        response = completion_with_backoff(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        generated_text = response.choices[0].text.strip()
    return generated_text

def construct_prompt(num_shot, examples):
    instruction = 'Create 5 new math word problems following the JSON format of the given examples.\n\nExample math word problems:\n\n'
    example_demonstrations = random.sample(examples, num_shot)
    prompt = instruction
    for i, line in enumerate(example_demonstrations):
        s = f"{i+1}): {line}\n"
        prompt += s
    prompt += '\n'
    prompt += f"Similar examples:\n\n{i+2}):"
    context = [{"role": 'user', "content": prompt}]
    return context


def main(output_dir, model, temperature, max_tokens, seed, num_shot, num_prompts, input_formula_dir, output_formula_dir):
    random.seed(seed)
    train_filename = 'processed_gsm8k/train.jsonl'
    examples = []
    
    with open(train_filename) as fin:
        for line in fin:
            #a = json.loads(line.strip())
            #question = a['question'].strip()
            #answer = a['answer'].strip()
            #examples.append((question, answer))
            examples.append(line.strip())

    prompts = []
    encoding = tiktoken.encoding_for_model(model)
    lens = []
    generated_examples = []
    os.makedirs(output_dir, exist_ok=True)
    prompt_filename = os.path.join(output_dir, 'prompts.jsonl')
    GET_PROMPTS = False
    if GET_PROMPTS:
        with open(prompt_filename, 'w') as fout:
            for _ in tqdm.tqdm(range(num_prompts)):
                #import pdb; pdb.set_trace()
                prompt = construct_prompt(num_shot, examples)
                text = prompt[-1]['content']
                #print (text)
                token_ids = encoding.encode(text)
                lens.append(len(token_ids))
                fout.write(json.dumps(prompt) + '\n')
    with open(prompt_filename) as fin:
        for line in fin:
            prompts.append(json.loads(line.strip()))
    prompts = prompts[:20000]
    #prompts = prompts[:10]


    #combinations = [(prompt, model, temperature, max_tokens) for prompt in prompts]
    #with Pool(10) as pool:
    #    #results = list(tqdm.tqdm(pool.imap(parallel_function, combinations), total=len(combinations)))
    #    pool.map(parallel_function, combinations)

    train_lines = []
    os.makedirs(output_formula_dir, exist_ok=True)
    for split in ['valid', 'test']:
        shutil.copy(os.path.join(input_formula_dir, f'src1_{split}.txt'), os.path.join(output_formula_dir, f'src1_{split}.txt'))
    with open(os.path.join(input_formula_dir, 'src1_train.txt')) as fin:
        for line in fin:
            train_lines.append(line.strip())

    WRITE_RESULTS = True
    if WRITE_RESULTS:
        start_time = time.time()  # Current time in seconds
        for prompt in prompts:
            completion = get_completion_with_cache(prompt, model, temperature, max_tokens)
            if completion is None:
                continue
            completion = f'{num_shot+1}): {completion.lstrip()}'
            #print (completion)
            for line in completion.split('\n'):
                items = line.strip().split(' ', 1)
                if len(items) == 2:
                    example = items[1]
                    flag = True
                    try:
                        d = json.loads(example)
                    except Exception as e:
                        print ('not parsable', example)
                        flag = False
                    if flag:
                        #import pdb; pdb.set_trace()
                        try:
                            question = d['question']
                            answer = d['answer']
                            formulas = re.findall(r'(<<[^>]+>>)', answer)
                            ans = answer.split('####')[-1].strip()
                            # Join the formulas with spaces
                            formulas_str = ' '.join(formulas)
                        except Exception as e:
                            continue
                        train_lines.append(question.strip() + '||' + formulas_str + f' #### {ans}')
        end_time = time.time()  # Current time in seconds after code execution
        execution_time_seconds = end_time - start_time
        execution_time_minutes = execution_time_seconds / 60
        print(f"Execution time: {execution_time_minutes:.2f} minutes")

        random.seed(1234)
        random.shuffle(train_lines)
        with open(os.path.join(output_formula_dir, 'src1_train.txt'), 'w') as fout:
            for line in train_lines:
                fout.write(line + '\n')




def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Augment")
    parser.add_argument("--model", type=str, default='gpt-4', choices=['gpt-3.5-turbo', 'gpt-4'], help="model to evaluate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=6000, help="Maximum number of tokens in the generated sentence")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_shot", type=int, default=5, help="")
    parser.add_argument("--num_prompts", type=int, default=100000, help="")
    parser.add_argument("--output_dir", type=str, default="output", help="Output folder")
    parser.add_argument("--input_formula_dir", type=str, default="data/math_scaffolding_formula", help="Output folder")
    parser.add_argument("--output_formula_dir", type=str, default="data/augmented_math_scaffolding_formula", help="Output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    main(output_dir=args.output_dir, \
         model=args.model, \
         temperature=args.temperature, \
         max_tokens=args.max_tokens, \
         seed=args.seed, \
         num_shot=args.num_shot, \
         num_prompts=args.num_prompts, \
         input_formula_dir=args.input_formula_dir, \
         output_formula_dir=args.output_formula_dir)
