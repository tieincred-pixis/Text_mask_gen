import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = smp.Unet('resnet34', classes=1, encoder_weights='imagenet').to(device)
model.load_state_dict(torch.load('finetune_weights_new/model_18.pt', map_location=device))
model.eval()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

image_folder = 'result_image_222.jpg'
out_save = 'poc_res'
single_test=True

if single_test:
    image = image_folder.split('/')[-1]
    img = Image.open(image_folder).convert('RGB')
    img = transform(img).unsqueeze(0)

    # Move the image to the device
    img = img.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(img)
        # Apply sigmoid function to normalize the output to [0, 1]
        output = torch.sigmoid(output)

    # Convert the output to a PIL image and save it
    output = output.cpu().squeeze().numpy()
    print(np.unique(output))
    output = (output > 0.9).astype('uint8') * 255
    output = Image.fromarray(output, mode='L')
    output.save(f'{out_save}/aa_{image}')

else:
    images = os.listdir(image_folder)

    for image in tqdm(images):
        fpth = os.path.join(image_folder, image)
        # Load the image
        img = Image.open(fpth).convert('RGB')
        img = transform(img).unsqueeze(0)

        # Move the image to the device
        img = img.to(device)

        # Make a prediction
        with torch.no_grad():
            output = model(img)
            # Apply sigmoid function to normalize the output to [0, 1]
            output = torch.sigmoid(output)

        # Convert the output to a PIL image and save it
        output = output.cpu().squeeze().numpy()
        print((np.unique(output)).min())
        output = (output > 0).astype('uint8') * 255
        output = Image.fromarray(output, mode='L')
        output.save(f'{out_save}/aaa_{image}' )