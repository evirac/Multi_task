from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def get_model(task):
    """
    Returns a preloaded model or pipeline for a given task.
    Automatically chooses CPU/GPU configuration and optimized models.
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[INFO] Using device: {device.upper()}")

    # --- SUMMARIZATION ---
    if task == "summarization":
        return pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if device != "cpu" else -1,
        )

    # --- SENTIMENT ANALYSIS ---
    elif task == "sentiment":
        # Use 3-class sentiment model (positive/neutral/negative)
        return pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=0 if device != "cpu" else -1,
        )

    # --- NEXT-WORD PREDICTION ---
    elif task == "nextword":
        tok = AutoTokenizer.from_pretrained("gpt2")
        mod = AutoModelForCausalLM.from_pretrained("gpt2")
        mod.to(device)
        return (tok, mod, device)

    # --- IMAGE GENERATION ---
    elif task == "imagegen":
        try:
            if device == "cuda":
                # full Stable Diffusion (fast with GPU)
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16
                )
                pipe.to("cuda")
                print("[INFO] Loaded Stable Diffusion v1.5 (GPU)")
                return pipe

            elif device == "mps":
                # Apple Silicon (M1/M2)
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32
                )
                pipe.to("mps")
                print("[INFO] Loaded Stable Diffusion (MPS)")
                return pipe

            else:
                # CPU fallback: use lightweight model
                from diffusers import AutoPipelineForText2Image
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sd-turbo",
                    torch_dtype=torch.float32
                )
                pipe.to("cpu")
                print("[INFO] Loaded SD-Turbo (CPU)")
                return pipe

        except Exception as e:
            print("[WARN] Local image model failed. Falling back to HF inference API.")
            from huggingface_hub import InferenceClient
            client = InferenceClient("stabilityai/stable-diffusion-2-1")
            return client

    else:
        raise ValueError(f"Unknown task: {task}")
