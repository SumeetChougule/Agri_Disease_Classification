import torch

from pathlib import Path
from torchinfo import summary
from src import engine, utils, model_builder
from src.data_preprocess import data_setup

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# directory paths to train and test images
train_dir = "data/train"
test_dir = "data/test"

# create a new model and transforms instance

effnetb2, effnetb2_transforms = model_builder.create_effnetb2_model(
    num_classes=20, device=device
)

# # Print EffNetB2 model summary (uncomment for full output)
# summary(
#     effnetb2,
#     input_size=(1, 3, 224, 224),
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"],
# )

# Load the data
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=effnetb2_transforms,
    batch_size=32,
)

# optimizer
optimizer = torch.optim.Adam(params=effnetb2.parameters(), lr=1e-3)

# loss fn
loss_fn = torch.nn.CrossEntropyLoss()

effnetb2_results = engine.train(
    model=effnetb2,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
)

utils.plot_loss_curves(results=effnetb2_results)

# save the model
utils.save_model(
    model=effnetb2,
    target_dir="models",
    model_name="pretrained_effnetb2_feature_extractor_crop_disease.pth",
)

# Count number of parameters in EffNetB2
effnetb2_total_params = sum(torch.numel(param) for param in effnetb2.parameters())

# Get the model size in bytes then convert to megabytes
pretrained_model_size = Path(
    "models/pretrained_effnetb2_feature_extractor_crop_disease.pth"
).stat().st_size // (
    1024 * 1024
)  # division converts bytes to megabytes (roughly)
print(f"Pretrained EffNetB2 feature extractor model size: {pretrained_model_size} MB")

# Create a dictionary with EffNetB2 statistics
effnetb2_stats = {
    "test_loss": effnetb2_results["test_loss"][-1],
    "test_acc": effnetb2_results["test_acc"][-1],
    "number_of_parameters": effnetb2_total_params,
    "model_size (MB)": pretrained_model_size,
}
