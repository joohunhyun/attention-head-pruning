from datasets import load_dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

def get_dataloaders(tokenizer, batch_size=16):
    dataset = load_dataset("glue", "mrpc")
    
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    val_loader = DataLoader(encoded_dataset["validation"], batch_size=batch_size, collate_fn=data_collator)

    return train_loader, val_loader
