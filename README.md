<h1 align="center"> Text-Removal: Unet Mask dedection </h1> 

## Usage

It's recommended to configure the environment using Anaconda. Python 3.8 + PyTorch 1.9.1 (or 1.9.0) + CUDA 11.1 + Detectron2 (v0.6) are suggested.

- ### Installation
```
conda create -n textRem python=3.8 -y
conda activate textRem
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python scipy timm shapely albumentations Polygon3
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install setuptools==59.5.0
git clone https://github.com/tieincred-pixis/Text_mask_gen.git
cd Text_mask_gen/TextRem
python setup.py build develop
pip install segmentation_models_pytorch
```

Text-Rem directory:
```
|- datasets
   |- syntext1
   |  |- train_images
   |  └─ train_poly_pos.json  # "pos" denotes the positional label form
   |- syntext2
   |  |- train_images
   |  └─ train_poly_pos.json
   |- mlt
   |  |- train_images
   |  └─ train_poly_pos.json
   |- totaltext
   |  |- test_images_rotate
   |  |- train_images_rotate
   |  |- test_poly.json
   |  |- test_poly_rotate.json
   |  |─ train_poly_ori.json
   |  |─ train_poly_pos.json
   |  |─ train_poly_rotate_ori.json
   |  └─ train_poly_rotate_pos.json
   |- ctw1500
   |  |- test_images
   |  |- train_images_rotate
   |  |- test_poly.json
   |  └─ train_poly_rotate_pos.json
   |- lsvt
   |  |- train_images
   |  └─ train_poly_pos.json
   |- art
   |  |- test_images
   |  |- train_images_rotate
   |  |- test_poly.json
   |  |─ train_poly_pos.json
   |  └─ train_poly_rotate_pos.json
   |- inversetext
   |  |- test_images
   |  └─ test_poly.json
   |- evaluation
   |  |- lexicons
   |  |- gt_totaltext.zip
   |  |- gt_ctw1500.zip
   |  |- gt_inversetext.zip
   |  └─ gt_totaltext_rotate.zip
```

The generation of positional label form is provided in `process_positional_label.py`

- ### Inference & Visualization
```
python demo/demo.py --config-file ${CONFIG_FILE} --input ${IMAGES_FOLDER_OR_ONE_IMAGE_PATH} --output ${OUTPUT_PATH} --opts MODEL.WEIGHTS <MODEL_PATH>
```
