import argparse
import json
from pathlib import Path
from tqdm import tqdm
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
from Prompts import format_for_create_safe
import warnings
import random
random.seed(42)
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cognitivecomputations/dolphin-2.9.4-llama3.1-8b',
                        help='Hugging Face model path or identifier.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs to use for model parallelism.')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1,
                        help='Number of pipeline stages (optional).')
    parser.add_argument('--input_dir', type=str, default='ydeng/duoguard-training')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--temp', type=float, default=0.9)
    parser.add_argument('--language', type=str, default='English')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for processing prompts')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Max tokens length for the model')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization fraction')
    parser.add_argument('--n_answers', type=int, default=8, help='Number of answers to generate per prompt')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_path = args.model
    input_dir = args.input_dir
    language = args.language
    batch_size = args.batch_size

    output_name = input_dir.split('/')[-1].replace('.json', '')

    # --- Load model and tokenizer ---
    logging.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization
        )
    except Exception as e:
        logging.error(f"Failed to load model or tokenizer: {e}")
        return

    # --- Sampling params ---
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=1.0,
        max_tokens=args.max_model_len,
        n=args.n_answers
    )

    # --- Load data ---
    logging.info(f"Loading data from {input_dir}...")
    try:
        data = load_dataset(input_dir, split=args.split, trust_remote_code=True, num_proc=8)
    except Exception as e:
        logging.error(f"Error loading input data: {e}")
        return

    # --- Shuffle data (seeded) ---
    logging.info("Shuffling data...")
    data = data.shuffle(seed=42)

    logging.info(f"Original dataset length: {len(data)}")
    filtered_data = []
    for example in data:
        prompt = example["prompt"]
        if prompt is not None:
            prompt = prompt.replace("*", "")
        if prompt:
            filtered_data.append(example)
    logging.info(f"Filtered dataset length: {len(filtered_data)}")

    prompts_all = [
        format_for_create_safe(example["prompt"], tokenizer, language=language)
        for example in filtered_data
    ]

    # --- Generate responses in batches ---
    logging.info(f"Generating responses in batches of size {batch_size}...")
    results = []
    try:
        for i in tqdm(range(0, len(prompts_all), batch_size)):
            batch_prompts = prompts_all[i:i+batch_size]
            batch_results = llm.generate(batch_prompts, sampling_params)
            results.extend([[r.text.replace("</s>", "").lstrip() for r in result.outputs] 
                            for result in batch_results])
    except Exception as e:
        logging.error(f"Error during prompt generation: {e}")
        return

    # --- Save results ---
    logging.info("Saving generated prompts...")
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{language}_safe_prompts_{output_name}.json"

    new_data = []
    for i, example in enumerate(filtered_data):
        new_example = example.copy()
        new_example["new_prompts"] = results[i]
        new_data.append(new_example)

    try:
        with open(output_file, "w") as f:
            json.dump(new_data, f)
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        return

    logging.info(f"Process completed. Example output for the first prompt: {results[0]}")


if __name__ == "__main__":
    main()
