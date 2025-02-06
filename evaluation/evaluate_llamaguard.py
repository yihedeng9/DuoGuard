import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import sys
import json
import os
from sklearn.metrics import f1_score
import argparse
import time  

model_ids = [
    "meta-llama/Llama-Guard-3-8B",
    "meta-llama/Meta-Llama-Guard-2-8B",
    "meta-llama/Llama-Guard-3-1B"
]
dtype = torch.bfloat16
device = 'cuda:5'

parser = argparse.ArgumentParser(description='Evaluate DuoGuard models.')
parser.add_argument('--language', type=str,
                    choices=['En', 'Fr', 'Es', 'De'],
                    default='En',
                    help='Language of the evaluation dataset')
args = parser.parse_args()
language = args.language

def moderate(chat, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

# Create logs directories if they don't exist
os.makedirs('./evaluation/logs', exist_ok=True)
os.makedirs('./evaluation/error_logs', exist_ok=True)

def add_class_xstest(example):
    example['class'] = 1 if example['type'].startswith('contrast_') else 0
    return example

def add_class_openai(example):
    example['class'] = example['moderation_flag']
    return example

def add_class_toxicchat(example):
    example['class'] = example['moderation_flag']
    return example

def preprocess_beavertail(example):
    example['class'] = 0 if example['is_safe'] else 1
    example['prompt'] = example['prompt'] + ' ' + example['response']
    return example

# Evaluation function
def evaluate(model, dataset, tokenizer):
    correct = 0
    total = len(dataset)

    predictions = []
    ground_truths = []

    # Start timing
    start_time = time.perf_counter()

    for item in tqdm(dataset, desc="Evaluating"):
        prompt = item['prompt']
        ground_truth = 'safe' if item['class'] == 0 else 'unsafe'
        prediction = moderate([{"role": "user", "content": prompt}], model, tokenizer).strip().split('\n')[0]

        predictions.append(prediction)
        ground_truths.append(ground_truth)

        if prediction == ground_truth:
            correct += 1

    # Ensure all GPU operations complete before recording end time
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_latency = total_time / total if total > 0 else 0.0

    accuracy = correct / total

    # FPR: proportion of unsafe ground-truth labeled safe
    unsafe_count = sum(gt == 'unsafe' for gt in ground_truths)
    fpr = (sum((p == 'safe' and gt == 'unsafe') 
               for p, gt in zip(predictions, ground_truths)) / unsafe_count
           if unsafe_count > 0 else 0.0)

    f1 = f1_score(ground_truths, predictions, pos_label='unsafe', average='binary')
    
    return {
        'FPR': fpr,
        'Accuracy': accuracy,
        'F-score': f1,
        'Average Latency (s)': avg_latency
    }

# Main loop for evaluating each checkpoint
for model_id in model_ids:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(examples['prompt'],
                         padding="max_length",
                         truncation=True,
                         max_length=2048,
                         return_tensors="pt")

    xstest_ds = load_dataset('csv',
                             data_files=f'./evaluation/eval_data_{language}/xstest_v2_completions_gpt4_gpteval.csv',
                             split='train')
    xstest_ds = xstest_ds.map(add_class_xstest).map(tokenize, batched=True, batch_size=100)

    openai_ds = load_dataset('json',
                             data_files=f'./evaluation/eval_data_{language}/openai.json',
                             split='train')
    openai_ds = openai_ds.map(add_class_openai).map(tokenize, batched=True, batch_size=100)

    toxicchat_ds = load_dataset('json',
                                data_files=f'./evaluation/eval_data_{language}/toxic_chat.json',
                                split='train')
    toxicchat_ds = toxicchat_ds.map(add_class_toxicchat).map(tokenize, batched=True, batch_size=100)

    rtp_ds = load_dataset('json',
                          data_files=f'./evaluation/eval_data_{language}/rtp_lx_prompt.json',
                          split='train')
    rtp_ds = rtp_ds.map(tokenize, batched=True, batch_size=100)

    xsafety_ds = load_dataset('json',
                              data_files=f'./evaluation/eval_data_{language}/XSafety.json',
                              split='train')
    xsafety_ds = xsafety_ds.map(tokenize, batched=True, batch_size=100)

    beavertails_ds = load_dataset('json',
                                  data_files=f'./evaluation/eval_data_{language}/beavertail_test.jsonl',
                                  split='train')
    beavertails_ds = beavertails_ds.map(preprocess_beavertail).map(tokenize, batched=True, batch_size=100)

    datasets = {
        'XSTest': xstest_ds,
        'OpenAI': openai_ds,
        'ToxicChat': toxicchat_ds,
        'BeaverTails': beavertails_ds,
        'RTP-LX': rtp_ds,
        'XSafety': xsafety_ds,
    }

    model_name = model_id.split('/')[-1]
    log_path = f'./evaluation/logs/eval_{language}_{model_name}.log'

    # Use a context manager to write logs
    with open(log_path, 'w') as log_file:
        print(f"Logging results to: {log_path}", file=sys.stdout)
        
        for name, dataset in datasets.items():
            results = evaluate(model, dataset, tokenizer)

            log_file.write(f"=== {name} ===\n")
            for k, v in results.items():
                log_file.write(f"{k}: {v:.3f}\n")
            log_file.write("\n")
            log_file.flush()
