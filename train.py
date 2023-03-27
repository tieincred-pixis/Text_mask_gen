import os
import torch
import shutil
import torchvision
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from data import Dataset
import numpy as np
from tqdm import tqdm
import sys
import wandb
from PIL import Image


finetune = False
model_file = 'finetune_weights/model_10.pt'
save_folder = 'finetune_weights'
if finetune:
    assert model_file.split('/')[0]!= save_folder, "loading model folder and save folder are same!"

if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)


# Initialize wandb
wandb.init(project='text_mask_extract')

# Set device to GPU if available, otherwise to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the encoder and decoder architecture of the model and move it to the device
model = smp.Unet('resnet34', classes=1, encoder_weights='imagenet').to(device)

start_epoch = 0
if finetune:
    print(f'fine tuning!! from model file {model_file}')
    start_epoch = 10
    model.load_state_dict(torch.load(model_file, map_location=device))
# Define your loss function and optimizer and move them to the device
criterion_mse = torch.nn.MSELoss().to(device)
criterion_bce = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Load your own dataset and split it into training, validation, and testing sets
train_dataset = Dataset('dataset_new','dataset_new_mask')
val_dataset = Dataset('eval','eval_mask')
test_dataset = Dataset('eval','eval_mask')

# Define your data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train the model
num_epochs = 10
steps = 1
for epoch in tqdm(range(num_epochs)):
    for images, masks in train_loader:
        # Move data to GPU
        # print(np.unique(masks[0]))
        # print('no to device')
        images, masks = images.to(device), masks.to(device)
        # print(np.unique(masks[0].cpu().numpy(), return_counts=True))
        # sys.exit()
        optimizer.zero_grad()
        outputs = model(images)
        # Convert the batch of images to a list of PIL Image objects
        output_batch_cpu = outputs.detach()
        output_batch_resized = F.interpolate(output_batch_cpu, size=(100, 100), mode='bilinear')
        # Log the images to wandb
        wandb.log({'output_batch': wandb.Image(output_batch_resized[0].cpu().numpy(), caption='Output Batch')})
        # Apply sigmoid function to normalize the output to [0, 1]
        output_mse = F.sigmoid(outputs)
        # Threshold the output to obtain a binary segmentation mask
        threshold = 0.5
        output_mse = (output_mse > threshold).float()
        loss_mse = criterion_mse(output_mse, masks)
        loss_bce = criterion_bce(outputs, masks)
        # print(masks.shape)
        # print(np.unique(outputs[0][0].detach().cpu().numpy()))
        # sys.exit()
        loss = loss_mse + loss_bce
        loss.backward()
        optimizer.step()
        if steps == 10000:
            scheduler.step()
            steps = 0
        # Log training loss to wandb
        wandb.log({'Training Loss MSE': loss_mse.item()})
        wandb.log({'Training Loss BCE': torch.sigmoid(loss_bce)*100})
        wandb.log({'Training Loss Total': loss.item()})
        # Get the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        # Log the current learning rate
        wandb.log({'learning_rate': current_lr})
        steps+=1

    if epoch % 2 == 0:
        torch.save(model.state_dict(), f'{save_folder}/model_{epoch+start_epoch}.pt')
    
    if epoch % 1 == 0:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                # Move data to GPU
                images, masks = images.to(device), masks.to(device)
                outputs_eval = model(images)
                output_batch_resized = F.interpolate(outputs_eval, size=(100, 100), mode='bilinear')
                # Log the images to wandb
                wandb.log({'output_batch_Eval': wandb.Image(output_batch_resized[0].cpu().numpy(), caption='Output Batch_eval')})
                loss_bce = criterion_bce(outputs_eval, masks)
                # Apply sigmoid function to normalize the output to [0, 1]
                output_mse = F.sigmoid(outputs_eval)
                # Threshold the output to obtain a binary segmentation mask
                threshold = 0.5
                output_mse = (output_mse > threshold).float()
                loss_mse = criterion_mse(output_mse, masks)
                loss = loss_bce + loss_mse
                total_loss += loss.item()

            avg_loss = total_loss / len(val_loader)
            print(f'Validation Loss: {avg_loss:.4f}')
            wandb.log({'Validation Loss': avg_loss})
        model.train()
    
    # Log epoch number every 10 epochs
    if epoch%2 == 0:
        wandb.log({'Epoch': epoch})
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Test the model on the testing set
total_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        # Move data to GPU
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion_bce(outputs, masks)
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')
