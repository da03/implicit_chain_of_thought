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
            #TODO     sep_id = input_ids_single.tolist().index(tokenizer.eos_token_id)
            #TODO     #sep_ids.append(sep_id)
            #TODO #assert all([item == sep_id for item in sep_ids])
            #TODO     tgt = input_ids_single[sep_id+1:]
            #TODO     max_new_tokens = tgt.size(0)+10
            #TODO     max_new_tokens = tgt[tgt.ne(tokenizer.eos_token_id)].size(0)+10
            #TODO     if not use_max:
            #TODO         max_new_tokens = 300
            #TODO     beam_output = model.generate(
            #TODO         input_ids=input_ids_single[:sep_id+1].unsqueeze(0),
            #TODO         max_new_tokens=max_new_tokens,
            #TODO         num_beams=beam_size,
            #TODO         early_stopping=True,
            #TODO         num_return_sequences=1,
            #TODO     )
            #TODO     ##src = input_ids_single[:sep_idx+1]
            #TODO     ##tgt = input_ids_single[sep_idx+1:]

            #TODO     ##sep_idx = tgt.tolist().index(tokenizer.eos_token_id)
            #TODO     ##tgt = tgt[:sep_idx]
            #TODO     ##tgt_text = tokenizer.decode(tgt)
            #TODO     ###ans = extract_answer(tgt_text)
            #TODO     ##ans = tgt_text.strip()


            #TODO     ##beam_output = model.generate(
            #TODO     ##    input_ids=src.view(1, -1),
            #TODO     ##    max_new_tokens=100,
            #TODO     ##    num_beams=5,
            #TODO     ##    early_stopping=True,
            #TODO     ##    num_return_sequences=1,
            #TODO     ##)
            #TODO     ##
            #TODO #import pdb; pdb.set_trace()
            #TODO #i = 0
            #TODO #for input_ids_single, beam_output_i in zip(input_ids, beam_output):
            #TODO     #tgt = tgt[:sep_id]
            #TODO     tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            #TODO     ans = extract_answer(tgt_text)
            #TODO     pred_text = tokenizer.decode(beam_output[0][sep_id+1:], skip_special_tokens=True)
            #TODO     if i == 0:
            #TODO         #print ("Output:\n" + 100 * '-')
            #TODO         #print (pred_text)
            #TODO         print ("\n" + 100 * '-')
            #TODO         print ('GT:', tokenizer.decode(input_ids_single, skip_special_tokens=True))
            #TODO         print ('Predicted:', pred_text)
            #TODO     pred_ans = extract_answer(pred_text) #.split()[-1] #extract_answer(pred_text)
            #TODO     #import pdb; pdb.set_trace()
            #TODO     if ans == pred_ans:
            #TODO         total_correct += 1
            #TODO     i += 1
            #TODO #break

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
    parser.add_argument('--eval_path', type=str, default=None)
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
    beam_size = 1
    #if 'fullcot' in args.train_path:
    #    dtype = 'bfloat16'
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
    trun = 1024
    if 'fullcot' in args.train_path:
        trun = 220
        if 'medium' in args.model:
            trun = 256
        if 'gpt' == args.model:
            trun = 256
    print (trun)
    train_dataset = CoTDataset(tokenizer, args.train_path, trun)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, trun)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataset = CoTDataset(tokenizer, args.test_path, trun)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    if args.eval_path is not None:
        eval_dataset = CoTDataset(tokenizer, args.eval_path, trun)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    else:
        eval_dataloader = None

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    #accuracy, word_accuracy, ppl = evaluate(model, val_dataloader, tokenizer, ctx, beam_size, use_max=True)
    #print (f'Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    accuracy, word_accuracy, ppl = evaluate(model, test_dataloader, tokenizer, ctx, beam_size, use_max=True)
    print (f'Test PPL: {ppl}. Test Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    if eval_dataloader is not None:
        accuracy, word_accuracy, ppl = evaluate(model, eval_dataloader, tokenizer, ctx, beam_size, use_max=True)
        print (f'Eval  PPL: {ppl}. Eval Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
    model.train()
    step = 0
    def save_model(model, tokenizer, model_dir):
        print ('saving', model_dir)
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
            print ('shape', input_ids.shape)
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
        #accuracy, word_accuracy, ppl = evaluate(model, val_dataloader, tokenizer, ctx, beam_size)
        #print (f'Epoch {epoch}. Validation PPL: {ppl}. Validation Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        accuracy, word_accuracy, ppl = evaluate(model, test_dataloader, tokenizer, ctx, beam_size)
        print (f'Test PPL: {ppl}. Test Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        if eval_dataloader is not None:
            accuracy, word_accuracy, ppl = evaluate(model, eval_dataloader, tokenizer, ctx, beam_size)
            print (f'Eval PPL: {ppl}. Eval Accuracy: {accuracy}. Word Accuracy: {word_accuracy}.')
        model.train()

if __name__ == "__main__":
    main()
