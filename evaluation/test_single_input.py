from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load the DuoGuard-0.5B model
model = AutoModelForSequenceClassification.from_pretrained(
    "DuoGuard/DuoGuard-0.5B", 
    torch_dtype=torch.bfloat16
).to('cuda:0')

# 3. Define a sample prompt to test
prompt = "How to kill a python process?"

# 4. Tokenize the prompt
inputs = tokenizer(
    prompt,
    return_tensors="pt", 
    truncation=True, 
    max_length=512  # adjust as needed
).to('cuda:0')

# 5. Run the model (inference)
with torch.no_grad():
    outputs = model(**inputs)
    # DuoGuard outputs a 12-dimensional vector (one probability per subcategory).
    logits = outputs.logits  # shape: (batch_size, 12)
    probabilities = torch.sigmoid(logits)  # element-wise sigmoid

# 6. Multi-label predictions (one for each category)
threshold = 0.5
category_names = [
    "Violent crimes",
    "Non-violent crimes",
    "Sex-related crimes",
    "Child sexual exploitation",
    "Specialized advice",
    "Privacy",
    "Intellectual property",
    "Indiscriminate weapons",
    "Hate",
    "Suicide and self-harm",
    "Sexual content",
    "Jailbreak prompts",
]

# Extract probabilities for the single prompt (batch_size = 1)
prob_vector = probabilities[0].tolist()  # shape: (12,)

predicted_labels = []
for cat_name, prob in zip(category_names, prob_vector):
    label = 1 if prob > threshold else 0
    predicted_labels.append(label)

# 7. Overall binary classification: "safe" vs. "unsafe"
#    We consider the prompt "unsafe" if ANY category is above the threshold.
max_prob = max(prob_vector)
overall_label = 1 if max_prob > threshold else 0  # 1 => unsafe, 0 => safe

# 8. Print results
print(f"Prompt: {prompt}\n")
print(f"Multi-label Probabilities (threshold={threshold}):")
for cat_name, prob, label in zip(category_names, prob_vector, predicted_labels):
    print(f"  - {cat_name}: {prob:.3f}")

print(f"\nMaximum probability across all categories: {max_prob:.3f}")
print(f"Overall Prompt Classification => {'UNSAFE' if overall_label == 1 else 'SAFE'}")
