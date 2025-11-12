import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score
import numpy as np


def compute_rouge_scores(predictions, references):
    """Compute average ROUGE-L, ROUGE-1, ROUGE-2 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        try:
            score = scorer.score(ref, pred)
            for key in scores.keys():
                scores[key].append(score[key].fmeasure)
        except Exception as e:
            print(f"[WARN] Skipping a sample due to error: {e}")

    # average scores
    avg_scores = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in scores.items()}
    return avg_scores


def compute_sentiment_accuracy(pred_labels, true_labels):
    """Compute accuracy for sentiment classification."""
    try:
        return float(accuracy_score(true_labels, pred_labels))
    except Exception as e:
        print(f"[ERROR] Failed to compute sentiment accuracy: {e}")
        return 0.0
