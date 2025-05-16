import torch
import torch.nn.functional as F

def compute_head_importance(model, dataloader, loss_fn, device):
    model.eval()
    head_importance = torch.zeros(model.config.num_hidden_layers, model.config.num_attention_heads).to(device)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Enable gradient computation for head_mask
        head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads).to(device)
        head_mask.requires_grad_(True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, head_mask=head_mask)
        loss = outputs.loss
        loss.backward()

        head_importance += head_mask.grad.abs().detach()

    # Normalize importance scores
    head_importance /= head_importance.sum()
    return head_importance

def prune_heads(model, head_importance, prune_ratio):
    num_layers = head_importance.shape[0]
    num_heads_per_layer = head_importance.shape[1]
    total_heads = num_layers * num_heads_per_layer
    num_prune = int(prune_ratio * total_heads)

    sorted_idx = torch.argsort(head_importance.view(-1))[:num_prune]

    heads_to_prune = {}
    for idx in sorted_idx:
        layer = (idx // num_heads_per_layer).item()
        head = (idx % num_heads_per_layer).item()
        heads_to_prune.setdefault(layer, []).append(head)

    model.bert.prune_heads(heads_to_prune)
    return model
