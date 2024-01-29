from flask import Flask, render_template, request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
#ABSTRACTIVE MODEL
model_name1 = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name1)
model1 = PegasusForConditionalGeneration.from_pretrained(model_name1).to(device)

# EXTRACTIVE MODEL
# model_name2 = "google/pegasus-cnn_dailymail"
# tokenizer2 = PegasusTokenizer.from_pretrained(model_name2)
# model2 = PegasusForConditionalGeneration.from_pretrained(model_name2).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarizationAbs', methods=["POST"])
def summarizeAbs():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model1.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)

    return render_template("output.html", data = {"summary": summary})

@app.route('/text-summarizationExt', methods=["POST"])
def summarizeExt():

    if request.method == "POST":

        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer2.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model2.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer2.decode(summary_[0], skip_special_tokens=True)


    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()

