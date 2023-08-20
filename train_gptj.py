import math
import torch
import sys
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import argparse
import os
import inspect
import tqdm
from data import CoTDataset, DataCollator
import logging
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

def evaluate(model, dataloader, tokenizer, ctx):
    total = 0
    word_correct = 0
    total_correct = 0
    total_loss = 0
    total_instances = 0
    for batch in tqdm.tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
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

        for i, input_ids_single in enumerate(input_ids):
            total_instances += 1
            sep_idx = input_ids_single.tolist().index(tokenizer.eos_token_id)
            src = input_ids_single[:sep_idx+1]
            tgt = input_ids_single[sep_idx+1:]

            sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
            tgt = tgt[:sep_idx]
            tgt_text = tokenizer.decode(tgt)
            ans = extract_answer(tgt_text)

            with torch.no_grad():
                with ctx:
                    beam_output = model.generate(
                        input_ids=src.view(1, -1),
                        max_new_tokens=100,
                        num_beams=5,
                        early_stopping=True,
                        num_return_sequences=1,
                    )
            
            pred_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
            if i == 0:
                print ("Output:\n" + 100 * '-')
                print (pred_text)
                sys.stdout.flush()
            pred_ans = extract_answer(pred_text)
            #import pdb; pdb.set_trace()
            if ans == pred_ans:
                total_correct += 1
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='gpt2')
    args = parser.parse_args()

    print (args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    print (ptdtype, dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).to(ptdtype)
    # TODO: maybe use pretrained model here?
    #model.apply(model._init_weights)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    collate_fn = DataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    # note: float16 data type will automatically use a GradScaler
    #compile = True # use PyTorch 2.0 to compile the model to be faster
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)


    accuracy, word_accuracy, ppl = evaluate(model, val_dataloader, tokenizer, ctx)
    print (f'Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    model.train()
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}") #TODO change epoch

        #model.save_pretrained("finetuned_gpt2")
        for batch in tqdm.tqdm(train_dataloader):
            #if epoch == 1:
            #    import pdb; pdb.set_trace()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
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
            loss.div(args.accumulate).backward()

            if step % args.accumulate == args.accumulate-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss = loss.item()
            ppl = math.exp(loss)
            if step % 100 == 0:
                print (f"Step: {step}. PPL: {ppl}. Accuracy: {accuracy}")
                sys.stdout.flush()
            step += 1
        accuracy, word_accuracy, ppl = evaluate(model, val_dataloader, tokenizer, ctx)
        print (f'Epoch {epoch}. Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        sys.stdout.flush()
        model.train()

if __name__ == "__main__":
    main()
