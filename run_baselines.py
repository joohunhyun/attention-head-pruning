import yaml
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from datasets import get_dataloaders
from train_eval import train, evaluate
from models.pruning_utils import compute_head_importance, prune_heads

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_name = os.getenv("MODEL_NAME", config["model_name"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config["num_labels"]).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loader, val_loader = get_dataloaders(tokenizer, batch_size=config["batch_size"])

optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
loss_fn = torch.nn.CrossEntropyLoss()

# Baseline training
train(model, train_loader, optimizer, loss_fn, device)
acc = evaluate(model, val_loader, device)
print(f"Accuracy before pruning: {acc:.4f}")

# HIS calculation & pruning
head_importance = compute_head_importance(model, train_loader, loss_fn, device)
model = prune_heads(model, head_importance, prune_ratio=config["prune_ratio"])

acc = evaluate(model, val_loader, device)
print(f"Accuracy after pruning: {acc:.4f}")
