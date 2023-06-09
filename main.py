import numpy as np
import gradio as gr
from keras.models import load_model

model = load_model("model.h5")
class_names = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

def classifier(img):
    img = np.array(img.resize((64,64)))
    return class_names[model.predict(img.reshape(1,64,64)).argmax()]

app = gr.Interface(fn=classifier, inputs=gr.Image(type="pil", image_mode="L"), outputs="label")
app.launch()
