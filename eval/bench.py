import time
from eval.metrics import compute_rouge
from models.registry import get_model

def benchmark_summarizer():
    summarizer = get_model("summarization")
    text = "Your test paragraph here..."
    start = time.time()
    result = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    end = time.time()
    print(f"Summary: {result}")
    print(f"Latency: {end - start:.2f}s")

if __name__ == "__main__":
    benchmark_summarizer()
