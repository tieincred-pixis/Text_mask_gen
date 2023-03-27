import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from cvt2bin import run_dptext_detr
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter

def enhance(image):
    contrast_enhancer = ImageEnhance.Contrast(image)
    enhanced_image = contrast_enhancer.enhance(2.0)
    return enhanced_image

def erode(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MinFilter(3))
    return image


def dilate(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MaxFilter(5))
    return image

def make_inference(modelT, modelM,config_file_path,input_image, enhance_contrast=False):
    out_name = input_image.split('/')[-1].split('.')[0]
    output_image_path = f'output/{out_name}'+'.png'
    maskout = f'masks/{out_name}'+'.png'
    try:
        run_dptext_detr(config_file_path, input_image, output_image_path, modelT, maskout)
    except Exception as e:
        print(e)
        pass
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = smp.Unet('resnet34', classes=1, encoder_weights='imagenet').to(device)
    model.load_state_dict(torch.load(modelM, map_location=device))
    model.eval()

    # Define the image transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image_folder = maskout
    image = image_folder.split('/')[-1]
    img = Image.open(image_folder).convert('RGB')
    if enhance_contrast:
        img = enhance(img)
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
    output = (output > 0.9).astype('uint8') * 255
    output = Image.fromarray(output, mode='L')
    eroded_image = erode(1,output)
    dilated_image = dilate(1,eroded_image)
    dilated_image.save('out.png')
    return dilated_image

if __name__ == '__main__':
    config_file_path = "configs/DPText_DETR/TotalText/R_50_poly.yaml"
    modelT = "weights/pretrain.pth"
    modelM = "weights/model_17.pt"
    input_image = "poc_images/6.jpg"
    make_inference(modelT, modelM, config_file_path, input_image)