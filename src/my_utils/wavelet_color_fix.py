'''
# --------------------------------------------------------------------------------
#   Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''

import torch
import math
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq

def fft_magnitude(x):
    """
    Compute FFT magnitude spectrum of an image tensor.
    Input:
        x: [B, C, H, W] in [-1, 1]
    Output:
        fft_mag: [B, C, H, W]
    """
    x = (x + 1.0) / 2.0  # scale to [0, 1]
    fft = torch.fft.fft2(x)
    fft_mag = torch.fft.fftshift(torch.abs(fft), dim=(-2, -1))
    return torch.log1p(fft_mag)

def compute_fft_loss(pred, target, loss_type='l1'):
    """
    Compute loss between FFT magnitude of predicted and target images.
    Inputs:
        pred, target: [B, C, H, W] in [-1, 1]
        loss_type: 'l1' or 'mse'
    Output:
        scalar loss
    """
    pred_mag = fft_magnitude(pred)
    target_mag = fft_magnitude(target)
    if loss_type == 'l1':
        return F.l1_loss(pred_mag, target_mag)
    elif loss_type == 'mse':
        return F.mse_loss(pred_mag, target_mag)
    else:
        raise ValueError("Unsupported loss_type. Use 'l1' or 'mse'.")

def compute_ndvi(image, red_index=0, nir_index=3, eps=1e-6):
    """
    image: Tensor of shape [B, C, H, W]
    red_index: channel index for RED band
    nir_index: channel index for NIR band
    """
    red = image[:, red_index, :, :]
    nir = image[:, nir_index, :, :]
    ndvi = (nir - red) / (nir + red + eps)
    return ndvi

def compute_ndvi_loss(pred, target, red_index=0, nir_index=3):
    pred_ndvi = (pred[:, nir_index] - pred[:, red_index]) / (pred[:, nir_index] + pred[:, red_index] + 1e-6)
    target_ndvi = (target[:, nir_index] - target[:, red_index]) / (target[:, nir_index] + target[:, red_index] + 1e-6)
    loss = F.l1_loss(pred_ndvi, target_ndvi, reduction='mean') 
    return loss

def compute_ndvi_weighted_fft_loss(pred, target, red_index=0, nir_index=3, threshold=0.05, eps=1e-6):
    """
    Compute FFT loss weighted by NDVI edge mask.
    pred, target: [B, C, H, W]
    red_index, nir_index: channel indices
    """
    # Step 1: compute NDVI maps from target (for mask)
    with torch.no_grad():
        ndvi = compute_ndvi(target, red_index=red_index, nir_index=nir_index, eps=eps)  # [B, H, W]
        grad_x = torch.abs(ndvi[:, :, 1:] - ndvi[:, :, :-1])
        grad_y = torch.abs(ndvi[:, 1:, :] - ndvi[:, :-1, :])
        grad = F.pad(grad_x, (0,1)) + F.pad(grad_y, (0,0,0,1))  # → [B, H, W]
        mask = (grad > threshold).float().unsqueeze(1)  # [B, 1, H, W]

    # Step 2: FFT loss
    pred_fft = torch.fft.fft2(pred, norm='ortho')
    target_fft = torch.fft.fft2(target, norm='ortho')

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    fft_diff = (pred_mag - target_mag) ** 2  # [B, C, H, W]
    loss = (fft_diff * mask).mean()  # weighted mean
    return loss


def save_rgb_composite(img_tensor, save_path, apply_brightness_enhance=True):
    """
    Save an RGB composite from a normalized [-1, 1] image tensor.
    - img_tensor: Tensor (C, H, W), normalized to [-1, 1]
    - save_path: Output path
    - apply_brightness_enhance: If True, adjusts brightness only for visualization
    """
    img_tensor = img_tensor.float()  # 🔧 convert to float32 for visualization
    img = (img_tensor * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
    img = img.clamp(0, 1)

    r = img[0]  
    g = img[1]  
    b = img[2]  

    rgb = torch.stack([r, g, b], dim=0)  # (3, H, W)

    if apply_brightness_enhance:
        mean_val = rgb.mean()
        target_mean = 0.35
        scale = target_mean / (mean_val + 1e-6)
        rgb = rgb * scale
        rgb = rgb.clamp(0, 1)

    rgb = (rgb * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_pil = Image.fromarray(rgb)
    img_pil.save(save_path)

def save_nir_swir_composite(img_tensor, save_path, apply_brightness_enhance=True):
    """
    Save NIR-SWIR composite from a normalized [-1, 1] image tensor.
    - img_tensor: Tensor (C, H, W), normalized to [-1, 1]
    - save_path: Output path
    - apply_brightness_enhance: If True, adjusts brightness only for visualization
    """
    img_tensor = img_tensor.float()  # 🔧 convert to float32 for visualization
    img = (img_tensor * 0.5) + 0.5  # Convert from [-1,1] to [0,1]
    img = img.clamp(0, 1)

    nir = img[0]    # NIR
    swir1 = img[1]  # SWIR1
    swir2 = img[2]  # SWIR2

    composite = torch.stack([nir, swir1, swir2], dim=0)  # (3, H, W)

    if apply_brightness_enhance:
        mean_val = composite.mean()
        target_mean = 0.35
        scale = target_mean / (mean_val + 1e-6)
        composite = composite * scale
        composite = composite.clamp(0, 1)

    composite = (composite * 255).byte().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img_pil = Image.fromarray(composite)
    img_pil.save(save_path)