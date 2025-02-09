import torch
import json

from torch import nn
from accelerate.state import PartialState
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from process_dataset import read_all_datasets
from transformers import DataCollatorWithPadding


class MultilabelCategoricalCrossentropyLoss(nn.Module):
    def __init__(self):
        super(MultilabelCategoricalCrossentropyLoss, self).__init__()

    def forward(self, outputs, labels):
        
        if hasattr(outputs, 'logits'):
            y_pred = outputs.logits
        else:
            y_pred = outputs

        y_true = labels
        y_pred_mod = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred_mod - y_true * 1e12
        y_pred_pos = y_pred_mod - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred_mod[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()  


# Custom Trainer to handle multi-label classification
class MultiLabelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = MultilabelCategoricalCrossentropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").float()

        outputs = model(**inputs)["logits"]
        loss = self.criterion(outputs, labels)

        return (loss, outputs) if return_outputs else loss


# Load configuration
with open("./classifier/config.json", "r") as config_file:
    config = json.load(config_file)

# Initialize model
model_id = config["model_id"]
dtype = getattr(torch, config["dtype"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map={"": PartialState().process_index},
    num_labels=12,
    attn_implementation="sdpa"
)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

tokenized_datasets = read_all_datasets(tokenizer, config["datasets"])

# Set up training arguments
training_params = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    optim=config["optim"],
    save_steps=config["save_steps"],
    logging_steps=config["logging_steps"],
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    fp16=config["fp16"],
    bf16=config["bf16"],
    tf32=config["tf32"],
    max_grad_norm=config["max_grad_norm"],
    max_steps=config["max_steps"],
    warmup_ratio=config["warmup_ratio"],
    group_by_length=config["group_by_length"],
    lr_scheduler_type=config["lr_scheduler_type"],
    ddp_find_unused_parameters=False,
    torch_compile=config["torch_compile"],
    report_to=config["report_to"],
    seed=config["seed"],
)

training_params = training_params.set_dataloader(num_workers=config["num_workers"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Initialize the trainer
trainer = MultiLabelTrainer(
    model=model,
    args=training_params,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Start training
trainer.train()
trainer.save_model(config["output_dir"] + '/' + config["output_model_name"])