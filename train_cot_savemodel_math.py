import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import argparse
import os
import sys
import tqdm
from data import CoTDataset, DataCollator
import logging
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
# or 
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        _, ans = text.strip().split('####', 1)
        ans = ans.strip().replace(',', '')
        return ans

def evaluate(model, dataloader, tokenizer, ctx, beam_size=5, use_max=False):
    with torch.no_grad():
        total = 0
        word_correct = 0
        total_correct = 0
        total_loss = 0
        total_instances = 0
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            with ctx:
                outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item() * labels[...,1:].ge(0).sum().item()

            labels_pred = logits.argmax(-1)
            correct = ((labels_pred[...,:-1] == labels[..., 1:]) * labels[..., 1:].ge(0)).sum().item()
            word_correct += correct
            total += labels[..., 1:].ge(0).sum().item()
            # TODO: generate and evaluate accuracy
            # activate beam search and early_stopping
            #import pdb; pdb.set_trace()

            #sep_ids = []
            for i, input_ids_single in enumerate(input_ids):
                total_instances += 1
                sep_id = input_ids_single.tolist().index(tokenizer.eos_token_id)
                #sep_ids.append(sep_id)
            #assert all([item == sep_id for item in sep_ids])
                tgt = input_ids_single[sep_id+1:]
                max_new_tokens = tgt.size(0)+10
                max_new_tokens = tgt[tgt.ne(tokenizer.eos_token_id)].size(0)+10
                if not use_max:
                    max_new_tokens = 300
                beam_output = model.generate(
                    input_ids=input_ids_single[:sep_id+1].unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    num_beams=beam_size,
                    early_stopping=True,
                    num_return_sequences=1,
                )
                ##src = input_ids_single[:sep_idx+1]
                ##tgt = input_ids_single[sep_idx+1:]

                ##sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
                ##tgt = tgt[:sep_idx]
                ##tgt_text = tokenizer.decode(tgt)
                ###ans = extract_answer(tgt_text)
                ##ans = tgt_text.strip()


                ##beam_output = model.generate(
                ##    input_ids=src.view(1, -1),
                ##    max_new_tokens=100,
                ##    num_beams=5,
                ##    early_stopping=True,
                ##    num_return_sequences=1,
                ##)
                ##
            #import pdb; pdb.set_trace()
            #i = 0
            #for input_ids_single, beam_output_i in zip(input_ids, beam_output):
                #tgt = tgt[:sep_id]
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text)
                pred_text = tokenizer.decode(beam_output[0][sep_id+1:], skip_special_tokens=True)
                if i == 0:
                    #print ("Output:\n" + 100 * '-')
                    #print (pred_text)
                    print ("\n" + 100 * '-')
                    print ('GT:', tokenizer.decode(input_ids_single, skip_special_tokens=True))
                    print ('Predicted:', pred_text)
                pred_ans = extract_answer(pred_text) #.split()[-1] #extract_answer(pred_text)
                #import pdb; pdb.set_trace()
                if ans == pred_ans:
                    total_correct += 1
                i += 1
            #break

        word_accuracy = word_correct / total
        accuracy = total_correct / total_instances
        loss = total_loss / total
        ppl = math.exp(loss)
    return accuracy, word_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/math_scaffolding/src1_train.txt')
    parser.add_argument('--val_path', type=str, default='data/math_scaffolding/src1_valid.txt')
    parser.add_argument('--test_path', type=str, default='data/math_scaffolding/src1_test.txt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--compile', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--save_model', type=str, default='model_nocot')
    args = parser.parse_args()

    print (args)

    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    if 'gpt2-xl' not in args.model:
        dtype = 'float32'
        beam_size = 5
    else:
        dtype = 'float32'
        beam_size = 1
    #beam_size = 1
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    print (ptdtype, dtype, beam_size)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).to(ptdtype)
    if args.compile == 1:
        model = torch.compile(model)
    else:
        print ('WARNING: no compile!')
    # TODO: maybe use pretrained model here?
    #model.apply(model._init_weights)
    use_fused = True 
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, **extra_args)
    #optimizer = AdamW(model.parameters(), lr=args.lr)

    collate_fn = DataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    accuracy, word_accuracy, ppl = evaluate(model, val_dataloader, tokenizer, ctx, beam_size, use_max=True)
    print (f'Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    model.train()
    step = 0
    def save_model(model, tokenizer, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    #import pdb; pdb.set_trace()
    for epoch in range(args.epochs):
        save_model(model, tokenizer, f'{args.save_model}/checkpoint_{epoch}_{args.lr}_{args.model}')
        print(f"Epoch {epoch}") #TODO change epoch

        #model.save_pretrained("finetuned_gpt2")
        for batch in tqdm.tqdm(train_dataloader):
            #if epoch == 1:
            #    import pdb; pdb.set_trace()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            #import pdb; pdb.set_trace()
            with ctx:
                outputs = model(input_ids=input_ids)
            #loss = outputs.loss
            logits = outputs.logits

            labels_pred = logits.argmax(-1)
            correct = ((labels_pred[...,:-1] == labels[...,1:]) * labels[...,1:].ge(0)).sum().item()
            total = labels[...,1:].ge(0).sum()
            accuracy = correct / total

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            ppl = math.exp(loss)
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Accuracy: {accuracy}")
                sys.stdout.flush()
            step += 1
        accuracy, word_accuracy, ppl = evaluate(model, val_dataloader, tokenizer, ctx, beam_size)
        print (f'Epoch {epoch}. Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        model.train()

if __name__ == "__main__":
    main()
