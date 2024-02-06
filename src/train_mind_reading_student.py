import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
import sys
import inspect
import tqdm
import logging
import random

from data_2 import CoTDataset, CoTDataCollator, extract_answer
from models.teacher import Teacher
from models.student import Student
from models.configuration_student import StudentConfig
from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

random.seed(1234)
torch.manual_seed(1234)
logging.disable(logging.WARNING)  # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    dataloader, tokenizer, ctx, teacher, student, delta, subset, max_new_tokens
):
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    for batch in tqdm.tqdm(dataloader):
        input_ids_all_1 = batch["input_ids_all_1"].to(device)
        input_ids_all_2 = batch["input_ids_all_2"].to(device)
        input_ids_nocot_1 = batch["input_ids_nocot_1"].to(device)
        input_ids_nocot_2 = batch["input_ids_nocot_2"].to(device)
        labels_nocot_1 = batch["labels_nocot_1"].to(device)
        labels_nocot_2 = batch["labels_nocot_2"].to(device)
        batch_size = input_ids_nocot_1.shape[0]
        with ctx:
            teacher_states_1 = teacher.extract_states(
                input_ids=input_ids_all_1, delta=delta, subset=subset
            )
            teacher_states_2 = teacher.extract_states(
                input_ids=input_ids_all_2, delta=delta, subset=subset
            )
            
            added_teacher_states = add_two_teacher_states(teacher_states_1,teacher_states_2)
            
            outputs = student.compute_loss(
                input_ids=torch.cat((input_ids_nocot_1, input_ids_nocot_2),1),
                labels=torch.cat((labels_nocot_1, labels_nocot_2),1),
                teacher_states=added_teacher_states,
            )
            
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()
        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        with ctx:
            beam_output_1 = student.generate(
                input_ids= input_ids_nocot_1,
                teacher_states=teacher_states_1,
                max_new_tokens=max_new_tokens,
            )
            
            beam_output_2 = student.generate(
                input_ids= input_ids_nocot_2,
                teacher_states=teacher_states_2,
                max_new_tokens=max_new_tokens,
            )
            
        # Evaluate
        sep_positions_1 = get_sep_position(input_ids_all_1, tokenizer.eos_token_id)
        sep_positions_2 = get_sep_position(input_ids_all_2, tokenizer.eos_token_id)
        
        for i, (input_ids_all_1_i, beam_output_1_i,input_ids_all_2_i, beam_output_2_i) in enumerate(
            zip(input_ids_all_1, beam_output_1 , input_ids_all_2, beam_output_2)
        ):
            sep_position_1 = sep_positions_1[i].item()
            tgt_1 = input_ids_all_1_i[sep_position_1 + 1 :]
            tgt_text_1 = tokenizer.decode(tgt_1, skip_special_tokens=True)
            ans_1 = extract_answer(tgt_text_1)
            pred_text_1 = tokenizer.decode(
                beam_output_1_i[0][sep_position_1 + 1 :], skip_special_tokens=True
            )
            pred_ans_1 = extract_answer(pred_text_1)
            
            sep_position_2 = sep_positions_2[i].item()
            tgt_2 = input_ids_all_2_i[sep_position_2 + 1 :]
            tgt_text_2 = tokenizer.decode(tgt_2, skip_special_tokens=True)
            ans_2 = extract_answer(tgt_text_2)
            pred_text_2 = tokenizer.decode(
                beam_output_2_i[0][sep_position_2 + 1 :], skip_special_tokens=True
            )
            pred_ans_2 = extract_answer(pred_text_2)
            
            if ans_1 + ' , ' + ans_2 == pred_ans_1 + ' , '+ pred_ans_2:
                total_correct += 1
            if i == 0:
                print(
                    f"Input: {tokenizer.decode(input_ids_all_1_i[:sep_position_1], skip_special_tokens=True)}"
                )
                print(
                    f"Input: {tokenizer.decode(input_ids_all_2_i[:sep_position_2], skip_special_tokens=True)}"
                )                
                
                print(f"Target: {tgt_text_1+ ' , ' + tgt_text_2}")
                print(f"Predicted: {pred_text_1 + ' , '+pred_text_2}")
                print("")
    accuracy = total_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return accuracy, token_accuracy, ppl

def add_two_teacher_states(teacher_states_1, teacher_states_2):
    added_teacher_states = []
    for t1,t2 in zip(teacher_states_1, teacher_states_2):
        added_teacher_states.append(torch.add(t1,t2))
    return added_teacher_states
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="train_models/4_by_4_mult/gpt2/teacher/checkpoint_0")
    parser.add_argument("--delta", type=str, default="dynamic")
    parser.add_argument("--train_path", type=str, default="data/4_by_4_mult/train_sample.txt")
    parser.add_argument("--val_path", type=str, default="data/4_by_4_mult/valid_sample.txt")
    parser.add_argument("--save_model", type=str, default="train_models/4_by_4_mult/gpt2/student_initial")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--base_model", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--subset",
        type=str,
        choices=["diagonal", "last_column", "top_row", "bottom_row", "first_column"],
        default="diagonal",
    )
    args = parser.parse_args()

    print(args)

    dtype = "float32"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    print(ptdtype, dtype, device)

    # Create Student
    config = StudentConfig(base_model=args.base_model)
    student = Student(config).to(device).to(ptdtype)

    # Load Teacher
    teacher = Teacher.from_pretrained(args.teacher).to(device).to(ptdtype)

    # Load data
    tokenizer = teacher.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 1024)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True
    )
    val_dataset = CoTDataset(tokenizer, args.val_path, 1024)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False
    )

    # Create Optimizer
    trainable_params = list(student.parameters())
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    teacher.eval()
    student.eval()  # to turn off dropout

    for p in teacher.parameters():
        p.requires_grad = False

    # Train
    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        for batch in tqdm.tqdm(train_dataloader):
            input_ids_all_1 = batch["input_ids_all_1"].to(device)
            input_ids_all_2 = batch["input_ids_all_2"].to(device)
            input_ids_nocot_1 = batch["input_ids_nocot_1"].to(device)
            input_ids_nocot_2 = batch["input_ids_nocot_2"].to(device)
            labels_nocot_1 = batch["labels_nocot_1"].to(device)
            labels_nocot_2 = batch["labels_nocot_2"].to(device)
            
            with ctx:
                with torch.no_grad():
                    teacher_states_1 = teacher.extract_states(
                        input_ids=input_ids_all_1, delta=args.delta, subset=args.subset
                    )

                    teacher_states_2 = teacher.extract_states(
                        input_ids=input_ids_all_2, delta=args.delta, subset=args.subset
                    )
                    
                added_teacher_states = add_two_teacher_states(teacher_states_1,teacher_states_2)   
                    
                outputs = student.compute_loss(
                    input_ids=torch.cat((input_ids_nocot_1, input_ids_nocot_2),1),
                    labels=torch.cat((labels_nocot_1, labels_nocot_2),1),
                    teacher_states=added_teacher_states,
                )
                
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            ppl = loss.exp().item()
            if step % 100 == 0:
                print(f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                sys.stdout.flush()
            step += 1
        accuracy, token_accuracy, ppl = evaluate(
            val_dataloader,
            tokenizer,
            ctx,
            teacher,
            student,
            args.delta,
            args.subset,
            args.max_new_tokens,
        )
        print(
            f"Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}."
        )
        student.save_pretrained(os.path.join(args.save_model, f"checkpoint_{epoch}"))


if __name__ == "__main__":
    main()
