import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
import logging
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from Prompts import format_for_rating
import warnings
warnings.filterwarnings("ignore")


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help='Hugging Face model path or identifier.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs used for tensor parallelism in vLLM.')
    parser.add_argument('--pipeline_parallel_size', type=int, default=1,
                        help='Number of pipeline parallel stages in vLLM.')
    parser.add_argument('--input_dir', type=str, default='outputs/French_prompts_meta-llama.json')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--language', type=str, default='French')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size for processing prompts')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Max tokens length for the model')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='Fraction of GPU memory to use.')
    parser.add_argument('--n_answers', type=int, default=8,
                        help='Number of answers to generate per prompt (not used in rating, but kept for consistency).')
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_path = args.model
    input_dir = args.input_dir
    language = args.language
    batch_size = args.batch_size

    # Derive an output name from the input file
    output_name = (
        input_dir.split('/')[-1]
        .replace('.json', '')
        .replace(f'{language}_prompts_', '')
    )

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Ensure the tokenizer has a pad token set
        tokenizer.pad_token = tokenizer.eos_token

        # Create the LLM instance with multi-GPU support
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

    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=1.0,
        max_tokens=args.max_model_len,
        best_of=1
    )

    logging.info(f"Loading data from {input_dir}...")
    try:
        with open(input_dir, "r") as f:
            data = json.load(f)
        logging.info(f"Data length: {len(data)}")
    except Exception as e:
        logging.error(f"Error loading input data: {e}")
        return
    
    # Prepare all rating prompts
    prompts_all = [
        format_for_rating(example["new_prompts"][i], tokenizer, language=language)
        for example in data
        for i in range(len(example["new_prompts"]))
    ]
    logging.info(f"Number of rating prompts: {len(prompts_all)}")

    # Generate responses (ratings) in batches
    logging.info(f"Generating responses in batches of {batch_size}...")
    results = []
    try:
        for i in tqdm(range(0, len(prompts_all), batch_size)):
            batch_prompts = prompts_all[i : i + batch_size]
            batch_results = llm.generate(batch_prompts, sampling_params)
            results.extend(
                [r.outputs[0].text.replace("</s>", "").lstrip() for r in batch_results]
            )
    except Exception as e:
        logging.error(f"Error during prompt generation: {e}")
        return

    # Clean up numeric ratings from the text responses
    cleaned_results = []
    for s in results:
        try:
            line = s.split("\n")[0].strip()
            cleaned_results.append(re.findall(r'\d+', line)[0])
        except:
            # Default to "5" if we can't parse a number
            # Usually it's when the LLM refuses to repond due to its own safeguard
            cleaned_results.append("5")

    logging.info(f"Results length: {len(results)}")

    # Create final output data
    logging.info("Attaching ratings and saving results...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    new_data = []
    index = 0
    for i, example in enumerate(data):
        # If there's no valid prompt, skip
        if example["prompt"] is None:
            continue
        new_example = example.copy()
        new_example["evaluations"] = []
        for j in range(len(example["new_prompts"])):
            new_example["evaluations"].append(cleaned_results[index])
            index += 1

        new_example["length_diff"] = [
            abs(len(example["prompt"]) - len(example["new_prompts"][j]))
            for j in range(len(example["new_prompts"]))
        ]
        new_data.append(new_example)

    out_file = output_dir / f"{language}_eval_{output_name}.json"
    try:
        with open(out_file, "w") as f:
            json.dump(new_data, f)
    except Exception as e:
        logging.error(f"Error saving output file: {e}")
        return

    logging.info(f"Process completed. Example output:\n{results[0]}")


if __name__ == "__main__":
    main()
