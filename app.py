import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "VridhiJain/roberta_bayesian"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    labels = ["Fox News", "NBC News"]
    return {labels[i]: float(probs[0][i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(lines=4, label="Enter News Headline"),
    outputs=gr.Label(label="Predicted Source"),
    title="News Source Classification",
    description="Classifies headlines as Fox News or NBC News using a fine-tuned RoBERTa model (Bayesian-optimized)."
)

demo.launch()

