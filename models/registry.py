from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch

def get_model(task):
    if task == "summarization":
        return pipeline("summarization", model="facebook/bart-large-cnn")
    elif task == "sentiment":
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    elif task == "nextword":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        mod = AutoModelForCausalLM.from_pretrained("gpt2")
        return (tok, mod)
    elif task == "imagegen":
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe
    else:
        raise ValueError("Unknown task")
