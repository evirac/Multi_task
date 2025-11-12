from datasets import load_metric
import torch
from transformers import AutoTokenizer

def compute_rouge(predictions, references):
    rouge = load_metric("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return {k: round(v.mid.fmeasure, 4) for k, v in results.items()}

def compute_accuracy(preds, labels):
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(preds)

def compute_perplexity(model, tokenizer, texts):
    # Small batch perplexity for language modeling tasks
    model.eval()
    total_loss, count = 0, 0
    with torch.no_grad():
        for t in texts:
            ids = tokenizer.encode(t, return_tensors="pt")
            loss = model(ids, labels=ids).loss.item()
            total_loss += loss
            count += 1
    return torch.exp(torch.tensor(total_loss / count)).item()
