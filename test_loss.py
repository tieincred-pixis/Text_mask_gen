import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from data import Dataset

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = smp.Unet('resnet34', classes=1, encoder_weights='imagenet').to(device)
model.load_state_dict(torch.load('saved_ckpt/model_10.pt', map_location=device))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_dataset = Dataset('my_dataset_test','masks_test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define your loss function and optimizer and move them to the device
criterion_mse = torch.nn.MSELoss().to(device)
criterion_bce = torch.nn.BCEWithLogitsLoss().to(device)

# Test the model on the testing set
total_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        # Move data to GPU
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss_mse = criterion_mse(outputs, masks)
        loss_bce = criterion_bce(outputs, masks)
        loss = loss_mse*100 + (torch.sigmoid(loss_bce)*100)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')