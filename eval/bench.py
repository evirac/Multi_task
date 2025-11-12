import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import json
from tqdm import tqdm
from models.registry import get_model
from eval.metrics import compute_rouge_scores, compute_sentiment_accuracy


# ======= PATHS =======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets")
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ======= LOAD MODELS =======
print("[INFO] Loading models...")
summarizer = get_model("summarization")
sentiment = get_model("sentiment")
print("[INFO] Models loaded successfully.")


# ======= SUMMARIZATION BENCHMARK =======
def evaluate_summarization():
    file_path = os.path.join(ASSETS_DIR, "summaries.csv")
    if not os.path.exists(file_path):
        print(f"[ERROR] summaries.csv not found at {file_path}")
        return None

    df = pd.read_csv(file_path)
    preds, refs = [], df["summary"].tolist()
    articles = df["article"].tolist()

    print("\nEvaluating summarization model...")
    for article in tqdm(articles):
        try:
            summary = summarizer(article, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            preds.append(summary)
        except Exception as e:
            print(f"[WARN] Failed to summarize: {e}")
            preds.append("")

    rouge = compute_rouge_scores(preds, refs)
    print("\n🔹 ROUGE Scores:")
    for k, v in rouge.items():
        print(f"{k.upper()}: {v:.4f}")

    return rouge


# ======= SENTIMENT BENCHMARK =======
def evaluate_sentiment():
    file_path = os.path.join(ASSETS_DIR, "sentiment.csv")
    if not os.path.exists(file_path):
        print(f"[ERROR] sentiment.csv not found at {file_path}")
        return None

    df = pd.read_csv(file_path)
    preds, truths = [], df["label"].tolist()
    texts = df["text"].tolist()

    print("\nEvaluating sentiment model...")
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    for text in tqdm(texts):
        try:
            res = sentiment(text)[0]
            preds.append(label_map.get(res["label"], res["label"]))
        except Exception as e:
            print(f"[WARN] Failed sentiment prediction: {e}")
            preds.append("Neutral")

    acc = compute_sentiment_accuracy(preds, truths)
    print(f"\n🔹 Sentiment Accuracy: {acc:.4f}")

    return acc


# ======= MAIN =======
if __name__ == "__main__":
    print("[INFO] Starting benchmark evaluation...")

    rouge = evaluate_summarization()
    acc = evaluate_sentiment()

    results = {
        "rouge_scores": rouge,
        "sentiment_accuracy": acc
    }

    metrics_path = os.path.join(OUTPUTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nMetrics saved to: {metrics_path}")
