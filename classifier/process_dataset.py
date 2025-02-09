from datasets import load_dataset, concatenate_datasets
import random
random.seed(42)

keys_order = [
    "C1(Violent Crimes)", "C2(Non-Violent Crimes)", "C3(Sex-Related Crimes)",
    "C4(Child Sexual Exploitation)", "C5(Specialized Advice)", "C6(Privacy)",
    "C7(Intellectual Property)", "C8(Indiscriminate Weapons)", "C9(Hate)",
    "C10(Suicide & Self-Harm)", "C11(Sexual Content)", "C12(Jailbreak Prompts)", 
]

def classes_to_labels(example):
    if example['moderation'] is not None:
        example["labels"] = [1 if example['moderation'][key] else 0 for key in keys_order]
    else:
        example["labels"] = [0] * len(keys_order)
    return example

def get_all_datasets(dataset_files):
    datasets = []
    for d in dataset_files:
        current_dataset = load_dataset(
            d, 
            split="train", 
            trust_remote_code=True, 
            num_proc=4
        )
        current_dataset = current_dataset.map(classes_to_labels, num_proc=4)
        current_dataset = current_dataset.filter(
            lambda x: x['prompt'] is not None and len(x['prompt']) > 2,
            num_proc=4
        )
        datasets.append(current_dataset)
    train_dataset = concatenate_datasets(datasets)
    return train_dataset

def read_all_datasets(tokenizer, dataset_files):
    train_dataset = get_all_datasets(dataset_files)
    train_dataset = train_dataset.shuffle()

    def preprocess_function(examples):
        return tokenizer(
            examples["prompt"], 
            truncation=True,
            max_length=512,
            padding=False
        )

    # Tokenize
    tokenized_datasets = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["prompt"],
        batch_size=1000,
        num_proc=4
    )

    return tokenized_datasets