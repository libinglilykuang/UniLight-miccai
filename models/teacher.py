import os

import torch
import torch.nn as nn
from monai.networks.nets import UNet


def get_teacher_model(device):
    """
    Initialize the Teacher Model (Standard 3D U-Net).
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)

    return model


def load_teacher_weights(model, weight_path, device):
    if not os.path.exists(weight_path):
        print(f"Teacher weights not found at {weight_path}")
        return model

    checkpoint = torch.load(weight_path, map_location=device)

    # Handle different saving formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    # Freeze teacher parameters to save memory and prevent accidental gradients
    for param in model.parameters():
        param.requires_grad = False

    print(f"Teacher weights loaded successfully from {weight_path}")
    return model