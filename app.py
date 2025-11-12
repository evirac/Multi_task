import gradio as gr
import torch
import yaml
import os
import json
import time
from models.registry import get_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ========= SAVE OUTPUTS ==========
def save_output(task, data):
    """Save outputs based on task type."""
    if not config.get("advanced", {}).get("save_outputs", False):
        return

    folder_map = {
        "image_generation": "outputs/images",
        "summarization": "outputs/summaries",
        "sentiment_analysis": "outputs/logs",
        "next_word_prediction": "outputs/logs",
    }

    rel_folder = folder_map.get(task, "outputs/other")
    folder = os.path.join(BASE_DIR, rel_folder)  # absolute path fix
    os.makedirs(folder, exist_ok=True)

    timestamp = int(time.time())

    try:
        if task == "image_generation" and isinstance(data, dict) and "image" in data:
            image = data["image"]
            image.save(os.path.join(folder, f"{timestamp}.png"))
        elif task == "summarization":
            with open(os.path.join(folder, f"{timestamp}.txt"), "w", encoding="utf-8") as f:
                f.write(data["summary"])
        else:
            with open(os.path.join(folder, f"{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        print(f"[SAVED] {task} output → {folder}")
    except Exception as e:
        print(f"[ERROR] Failed to save {task} output: {e}")



# ========= CONFIG LOADER ==========
CONFIG_PATH = "configs/app.yaml"

def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
print("[INFO] Loaded configuration from", CONFIG_PATH)

# ========= MODEL LOADING ==========
summarizer = sentiment = tokenizer = gpt2 = device = imagegen = None

if config['app']['enable_tabs'].get('summarization', False):
    summarizer = get_model("summarization")

if config['app']['enable_tabs'].get('sentiment_analysis', False):
    sentiment = get_model("sentiment")

if config['app']['enable_tabs'].get('next_word_prediction', False):
    tokenizer, gpt2, device = get_model("nextword")

if config['app']['enable_tabs'].get('image_generation', False):
    imagegen = get_model("imagegen")


# ========= TASK FUNCTIONS =========

# ---- Summarization ----
def summarize_text(text):
    if not text.strip():
        return "⚠️ Please enter some text to summarize."

    limits = config.get('limits', {})
    max_chunk_words = limits.get('summarization_max_words', 400)
    min_length = limits.get('summarization_min_length', 40)
    max_length = limits.get('summarization_max_length', 150)

    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]

    summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )[0]['summary_text']
        summaries.append(summary)

    final_summary = " ".join(summaries).strip()

    # Save output
    save_output("summarization", {"input": text, "summary": final_summary})
    return final_summary


# ---- Sentiment Analysis ----
def analyze_sentiment(text):
    if not text.strip():
        return "⚠️ Please enter text."

    result = sentiment(text)[0]
    label = result['label']
    score = result['score']

    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    label_readable = label_map.get(label, label)

    output_text = f"{label_readable} ({score:.2f})"

    # Save output
    save_output("sentiment_analysis", {"input": text, "output": output_text})
    return output_text


# ---- Next-Word Prediction ----
def predict_next_word(prompt):
    if not prompt.strip():
        return "⚠️ Please enter a prompt."

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = gpt2.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    text_out = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Save output
    save_output("next_word_prediction", {"input": prompt, "output": text_out})
    return text_out


# ---- Image Generation ----
def generate_image(prompt):
    if not prompt.strip():
        return None

    try:
        if hasattr(imagegen, "text_to_image"):  # Hugging Face API
            img = imagegen.text_to_image(prompt)
        else:  # Diffusers pipeline
            with torch.no_grad():
                result = imagegen(prompt)
                if hasattr(result, "images"):
                    img = result.images[0]
                elif isinstance(result, dict) and "images" in result:
                    img = result["images"][0]

        # Save output
        save_output("image_generation", {"prompt": prompt, "image": img})
        return img

    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        return None


# ========= GRADIO UI ==========
with gr.Blocks(theme=config['app'].get('theme', 'default')) as demo:
    gr.Markdown(f"# 🧩 {config['app'].get('title', 'Multi-Task AI App')}")

    if config['app']['enable_tabs'].get('image_generation', False):
        with gr.Tab("🖼️ Image Generation"):
            prompt = gr.Textbox(label="Prompt")
            out = gr.Image(label="Generated Image")
            btn = gr.Button("Generate")
            btn.click(generate_image, prompt, out)

    if config['app']['enable_tabs'].get('summarization', False):
        with gr.Tab("📰 Summarization"):
            txt = gr.Textbox(lines=5, label="Enter Text")
            out_sum = gr.Textbox(label="Summary")
            btn2 = gr.Button("Summarize")
            btn2.click(summarize_text, txt, out_sum)

    if config['app']['enable_tabs'].get('sentiment_analysis', False):
        with gr.Tab("💬 Sentiment Analysis"):
            s_in = gr.Textbox(label="Enter Text")
            s_out = gr.Textbox(label="Sentiment Result")
            s_btn = gr.Button("Analyze")
            s_btn.click(analyze_sentiment, s_in, s_out)

    if config['app']['enable_tabs'].get('next_word_prediction', False):
        with gr.Tab("🔤 Next-Word Prediction"):
            p_in = gr.Textbox(label="Prompt")
            p_out = gr.Textbox(label="Continuation")
            p_btn = gr.Button("Predict")
            p_btn.click(predict_next_word, p_in, p_out)

demo.launch()
