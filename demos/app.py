import os
import gradio as gr
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class name
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
    "jute",
    "maize",
    "rice",
    "sugarcane",
    "wheat",
]


# Model and transforms
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=len(class_names), device="cpu"
)

# Load saved weights
effnetb2.load_state_dict(
    torch.load(
        f="cropguardian/pretrained_effnetb2_feature_extractor_crop_disease.pth",
        map_location="cpu",
    )
)

# Predict function


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into eval mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    predict_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dic and pred time
    return predict_labels_and_probs, pred_time


# Gradio app

# Create title and description
title = "CropGuardian"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of various diseases affecting tomatoes, potatoes, peppers, as well as different crop classes like wheat, jute, rice, maize, and sugarcane."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# create the Gradio demo
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=3, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
)

# Launch the demo
demo.launch()
