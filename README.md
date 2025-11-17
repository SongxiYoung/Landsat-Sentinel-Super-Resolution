# Landsat-Sentinel-Super-Resolution

## 1. Conda Environment Setup
Option1:
```
conda env create -f environment.yml
```

Option2:
Follow the [PiSA-SR installation steps](https://github.com/csslc/PiSA-SR/tree/main).

## 2. Download Pretrained Models

- Download the pretrained SD-2.1-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).
- Download the RAM model from [HuggingFace](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) and save the model to "./src/ram_pretrain_model/".

## 3. Create Data Path File
```
python ./scripts/get_path.py
```

## 3. Model Training
```
bash ./scripts/train/train_lssr.sh
```

## 4. Model Inference
```
bash ./scripts/test/test_lssr.sh
```

## Citations:
```
PiSA-SR

LSSR data

LSSR model
```
