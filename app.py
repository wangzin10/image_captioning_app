from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import torch
from model_loader import Trainer
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import os

from torchvision import transforms
from types import SimpleNamespace
from pathlib import Path


app = Flask(__name__)

# Load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_config = SimpleNamespace(
    vocab_size = 50_257,
    embed_dim = 768,
    num_heads = 12,
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)
train_config = SimpleNamespace(
    epochs = 10,
    freeze_epochs_gpt = 1,
    freeze_epochs_all = 2,
    lr = 1e-4,
    device = device, 
    model_path = Path('D:/HK_personal project/project_1_imagecaptioning/best_model'),
    batch_size = 32
)
class DummyDL:
    def __len__(self): return 1
    def __iter__(self): 
        yield None

dls = (DummyDL(), DummyDL())

trainer = Trainer(model_config=model_config,
                  train_config=train_config,
                  dls=dls)

trainer.load_best_model()



# Load Translator
model_name = "Helsinki-NLP/opus-mt-en-vi"
translator_tokenizer = MarianTokenizer.from_pretrained(model_name)
translator_model = MarianMTModel.from_pretrained(model_name).to(device)

def translate_to_vietnamese(text):
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True).to(device)
    outputs = translator_model.generate(**inputs)
    return translator_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    # Generate caption
    caption_en = trainer.generate_caption(
            image=image,
            max_tokens=50,
            temperature=1.0,
            deterministic=True
        )
    caption_vi = translate_to_vietnamese(caption_en)

    # Save TTS
    tts = gTTS(caption_vi, lang='vi')
    tts.save("static/caption.mp3")

    return {
        "caption_en": caption_en,
        "caption_vi": caption_vi,
        "audio_url": "/static/caption.mp3"
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

