import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import sys
import json
import os
import argparse
import time

### Important ###
# Change model tokenizer to the corresponding tokenizer #
tokenizers = {
              "DuoGuard/DuoGuard-0.5B": "Qwen/Qwen2.5-0.5B",
              "DuoGuard/DuoGuard-1.5B-transfer": "Qwen/Qwen2.5-1.5B",
              "DuoGuard/DuoGuard-1B-Llama-3.2-transfer": "meta-llama/Llama-3.2-1B",
              }

OUTPUT_ERROR_LOGS = False
DEVICE = 'cuda:5'
CHECKPOINT = "DuoGuard/DuoGuard-1.5B-transfer"

parser = argparse.ArgumentParser(description='Evaluate DuoGuard models.')
parser.add_argument('--language', type=str, choices=['En', 'Fr', 'Es', 'De'], default='En', help='Language of the evaluation dataset')
args = parser.parse_args()
language = args.language

# Create logs directory if it doesn't exist
os.makedirs('evaluation/logs', exist_ok=True)
os.makedirs('evaluation/error_logs', exist_ok=True)

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(tokenizers[CHECKPOINT]) 
tokenizer.pad_token = tokenizer.eos_token

# Load datasets
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

def tokenize(examples):
    return tokenizer(
        examples['prompt'],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )

xstest_ds = load_dataset('csv', data_files=f'./evaluation/eval_data_{language}/xstest_v2_completions_gpt4_gpteval.csv', split='train')
xstest_ds = xstest_ds.map(add_class_xstest).map(tokenize, batched=True, batch_size=100)

openai_ds = load_dataset('json', data_files=f'./evaluation/eval_data_{language}/openai.json', split='train')
openai_ds = openai_ds.map(add_class_openai).map(tokenize, batched=True, batch_size=100)

toxicchat_ds = load_dataset('json', data_files=f'./evaluation/eval_data_{language}/toxic_chat.json', split='train')
toxicchat_ds = toxicchat_ds.map(add_class_toxicchat).map(tokenize, batched=True, batch_size=100)

rtp_ds = load_dataset('json', data_files=f'./evaluation/eval_data_{language}/rtp_lx_prompt.json', split='train')
rtp_ds = rtp_ds.map(tokenize, batched=True, batch_size=100)

xsafety_ds = load_dataset('json', data_files=f'./evaluation/eval_data_{language}/XSafety.json', split='train')
xsafety_ds = xsafety_ds.map(tokenize, batched=True, batch_size=100)

beavertails_ds = load_dataset('json', data_files=f'./evaluation/eval_data_{language}/beavertail_test.jsonl', split='train')
beavertails_ds = beavertails_ds.map(preprocess_beavertail).map(tokenize, batched=True, batch_size=100)

datasets = {
    'XSTest': xstest_ds,
    'OpenAI': openai_ds,
    'ToxicChat': toxicchat_ds,
    'BeaverTails': beavertails_ds,
    'RTP-LX': rtp_ds,
    'XSafety': xsafety_ds
}

# Evaluation function with average latency calculation
def evaluate(model, dataset, threshold=0.5, dataname='default', checkpoint_name='default', output_error_logs=True):
    probabilities = []
    predictions = []
    fp_data, fn_data = [], []
    fp_prob, fn_prob = [], []

    # Start timing
    start_time = time.perf_counter()

    with torch.no_grad():
        for example in tqdm(dataset):
            input_ids = torch.tensor([example['input_ids']]).to(DEVICE)
            attention_mask = torch.tensor([example['attention_mask']]).to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probability = torch.max(torch.sigmoid(outputs.logits)).item()
            probabilities.append(probability)
            predictions.append(1 if probability > threshold else 0)

    # GPU synchronization to ensure all CUDA kernels finish before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_latency = total_time / len(dataset) if len(dataset) > 0 else 0.0

    # Calculate metrics
    tp, fp, tn, fn = 0, 0, 0, 0
    for index, (pred, label) in enumerate(zip(predictions, dataset['class'])):
        if label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
            if output_error_logs:
                example = dataset[index]
                fp_data.append(example['prompt'])
                fp_prob.append(probabilities[index])
        elif label == 1 and pred == 0:
            fn += 1
            if output_error_logs:
                example = dataset[index]
                # Removing large fields to save memory if needed
                del example['input_ids']
                del example['attention_mask']
                fn_data.append(example)
                fn_prob.append(probabilities[index])
        else:
            tp += 1

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    auprc = average_precision_score(dataset['class'], probabilities)

    # Error logs & histograms
    if output_error_logs:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['axes.labelsize'] = 16
        plt.rcParams.update({'figure.figsize': [6.0, 4.5]})

        with open(f'./evaluation/error_logs/fp_{dataname}_{checkpoint_name}.json', 'w') as f:
            json.dump(fp_data, f, indent=4)
        with open(f'./evaluation/error_logs/fn_{dataname}_{checkpoint_name}.json', 'w') as f:
            json.dump(fn_data, f, indent=4)

        plt.hist(fp_prob, bins=25, alpha=0.75, color='red', edgecolor='black')
        plt.title(f'False Positive - {dataname}', fontsize=18)
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'./evaluation/error_logs/fp_prob_histogram_{dataname}_{checkpoint_name}.pdf', bbox_inches='tight')
        plt.close()

        plt.hist(fn_prob, bins=25, alpha=0.75, color='blue', edgecolor='black')
        plt.title(f'False Negative - {dataname}', fontsize=18)
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'./evaluation/error_logs/fn_prob_histogram_{dataname}_{checkpoint_name}.pdf', bbox_inches='tight')
        plt.close()

    return {
        'FPR': fpr,
        'Precision': precision,
        'Recall': recall,
        'F-score': f_score,
        'AUPRC': auprc,
        'Average Latency (s)': avg_latency
    }

checkpoint_name = CHECKPOINT.split('/')[-1]
log_file = open(f'./evaluation/logs/eval_{language}_{checkpoint_name}.log', 'w')
sys.stdout = log_file

finetuned_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, torch_dtype=torch.bfloat16).to(DEVICE)

for name, dataset in datasets.items():
    results = evaluate(
        finetuned_model,
        dataset,
        dataname=name,
        checkpoint_name=checkpoint_name,
        output_error_logs=OUTPUT_ERROR_LOGS
    )

    print(f'### {name} ###')
    print('--- Results ---')
    [print(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}') for k, v in results.items()]
    print()
    print()

log_file.close()
