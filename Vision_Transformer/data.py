from src import utils
from torchvision import transforms
from src.data_preprocess import data_setup

# directory paths to train and test images
train_dir = "data/train"
test_dir = "data/test"

# create image size (from Table 3 of ViT paper)
IMG_SIZE = 224

# transform pipeline
manual_transforms = transforms.Compose(
    [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
)

# Set the batch size
BATCH_SIZE = (
    32  # this is lower than the ViT paper but it's because we're starting small
)


# Load the data
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=32,
)
