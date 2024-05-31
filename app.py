import torch
import random
import gradio as gr

from pathlib import Path
from PIL import Image
from src import model_builder
from typing import List, Dict, Tuple
from timeit import default_timer as timer
from src.data_preprocess import data_setup


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


# directory paths to train and test images
train_dir = "data/train"
test_dir = "data/test"


MODEL_SAVE_PATH = "models/pretrained_effnetb2_feature_extractor_crop_disease.pth"

# Instantiate a new instance of our model (this will be instantiated with random weights)
trained_effnetb2, effnetb2_transforms = model_builder.create_effnetb2_model(
    num_classes=20, device=device
)

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
trained_effnetb2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Load the data
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=effnetb2_transforms,
    batch_size=32,
)

trained_effnetb2.to("cpu")
# check the device
next(iter(trained_effnetb2.parameters())).device


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    trained_effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(trained_effnetb2(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


test_data_paths = list(Path(test_dir).glob("*/*.jpeg"))

random_image_path = random.sample(test_data_paths, k=1)[0]

image = Image.open(random_image_path)
print(f"[INFO] Predicting on image at path: {random_image_path}\n")

pred_dict, pred_time, pred_class = predict(img=image)
print(f"Prediction label and probability dictionary: \n{pred_dict}")
print(f"Prediction time: {pred_time} seconds")
print(f"Predicted label {pred_class}")

img_infer = Image.open("test.jpg")

pred_dict, pred_time, pred_class = predict(img=img_infer)


# Create a list of example inputs to our Gradio demo
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

# Create title, description and article strings
title = "CropGuardian"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of various diseases affecting tomatoes, potatoes, peppers, as well as different crop classes like wheat, jute, rice, maize, and sugarcane."

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
demo.launch(debug=False, share=True)
