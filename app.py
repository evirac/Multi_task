import gradio as gr
from models.registry import get_model
import torch

summarizer = get_model("summarization")
sentiment = get_model("sentiment")
tokenizer, gpt2 = get_model("nextword")
img_pipe = get_model("imagegen")

def summarize(text):
    return summarizer(text, max_length=120, min_length=25, do_sample=False)[0]['summary_text']

def analyze_sentiment(text):
    result = sentiment(text)[0]
    return f"{result['label']} ({result['score']:.2f})"

def predict_next_word(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = gpt2.generate(input_ids, max_new_tokens=20, do_sample=True, top_p=0.95, top_k=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_image(prompt):
    image = img_pipe(prompt).images[0]
    return image

with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🧩 Multi-Task AI App")
    with gr.Tab("🖼 Image Generation"):
        prompt = gr.Textbox(label="Prompt")
        out = gr.Image()
        btn = gr.Button("Generate")
        btn.click(generate_image, prompt, out)

    with gr.Tab("📰 Summarization"):
        txt = gr.Textbox(lines=5, label="Enter Text")
        out_sum = gr.Textbox(label="Summary")
        btn2 = gr.Button("Summarize")
        btn2.click(summarize, txt, out_sum)

    with gr.Tab("💬 Sentiment Analysis"):
        s_in = gr.Textbox(label="Enter Text")
        s_out = gr.Textbox(label="Sentiment")
        s_btn = gr.Button("Analyze")
        s_btn.click(analyze_sentiment, s_in, s_out)

    with gr.Tab("🔤 Next-Word Prediction"):
        p_in = gr.Textbox(label="Prompt")
        p_out = gr.Textbox(label="Continuation")
        p_btn = gr.Button("Predict")
        p_btn.click(predict_next_word, p_in, p_out)

demo.launch()
