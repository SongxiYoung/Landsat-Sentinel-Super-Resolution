import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm


from lssr import LSSR_eval
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from src.my_utils.wavelet_color_fix import save_rgb_composite, save_nir_swir_composite
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from torchvision import models
import torch.nn.functional

from src.datasets.dataset_lssr import PairedSROnlineTxtDataset, save_ndvi_map
from torch.utils.data import DataLoader

import glob
import math



def calculate_sam(gt_01, pred_01, eps=1e-8, mask_thr=1e-6, ignore_border=8, trim_top_p=0.01):
    """
    gt_01, pred_01: (1, C, H, W) in [0,1]
    """
    # 4D
    if gt_01.dim() == 3:  gt_01 = gt_01.unsqueeze(0)
    if pred_01.dim() == 3: pred_01 = pred_01.unsqueeze(0)

    # type
    gt_01   = gt_01.to(dtype=torch.float32).contiguous()
    pred_01 = pred_01.to(dtype=torch.float32).contiguous()

    # clip
    if ignore_border and ignore_border > 0:
        H, W = gt_01.shape[-2], gt_01.shape[-1]
        ib = min(ignore_border, max(H // 2 - 1, 0), max(W // 2 - 1, 0))
        if ib > 0:
            gt_01   = gt_01[:, :, ib:-ib, ib:-ib]
            pred_01 = pred_01[:, :, ib:-ib, ib:-ib]

    # SAM
    B, C, H, W = gt_01.shape
    xv = gt_01.reshape(B, C, -1).permute(0, 2, 1).contiguous()   # (B,HW,C)
    yv = pred_01.reshape(B, C, -1).permute(0, 2, 1).contiguous() # (B,HW,C)

    dot = (xv * yv).sum(dim=-1)          # (B,HW)
    nx  = torch.linalg.norm(xv, dim=-1)
    ny  = torch.linalg.norm(yv, dim=-1)

    cos = dot / (nx * ny + eps)
    cos = torch.clamp(cos, -1.0, 1.0)
    ang = torch.acos(cos) * (180.0 / math.pi)  # degrees

    # mask
    valid = (nx > mask_thr) & (ny > mask_thr)
    ang = torch.where(valid, ang, torch.nan)

    # NaN
    vals = ang.reshape(B, -1).squeeze(0)
    vals = vals[~torch.isnan(vals)]
    if vals.numel() == 0:
        return torch.tensor(float('nan'), device=ang.device)

    if trim_top_p is not None and trim_top_p > 0:
        q = torch.quantile(vals, 1 - trim_top_p)
        vals = vals[vals <= q]
        if vals.numel() == 0:
            return torch.tensor(float('nan'), device=ang.device)

    return vals.mean()


def compute_ndvi(img_tensor):
    """
    Compute NDVI from a 6-band image tensor (C, H, W) normalized to [-1, 1].
    Bands: [B2, B3, B4, B8, B11, B12] → [0, 1, 2, 3, 4, 5]
    """
    img_tensor = img_tensor * 0.5 + 0.5  # Convert to [0, 1]
    red = img_tensor[2]  # Band 4
    nir = img_tensor[3]  # Band 8
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6  # M

def pisa_sr(args):
    model = LSSR_eval(args)
    model.set_eval()

    total_params = model.count_parameters(model)
    print(f"LSSR Params: Total = {total_params:.2f}B")

    print("\n==== Custom Modules Parameter Count (in Millions) ====")
    for name, module in [
        ("DEM Encoder", model.dem_encoder),
        ("LC Encoder", model.lc_encoder),
        ("Month Encoder", model.month_encoder),
        ("Latent Project In", model.latent_project_in),
        ("Latent Project Out", model.latent_project_out),
        ("Cross Attention", model.cross_attn),
    ]:
        total, trainable = count_parameters(module)
        print(f"{name:20s} | Total: {total:.3f}M | Trainable: {trainable:.3f}M")

    lpips_loss = lpips.LPIPS(net='vgg').cuda()

    resnet = models.resnet18(pretrained=True).eval().cuda()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:6])
    # print("DEBUG")
    dataset = PairedSROnlineTxtDataset(split="test", args=args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)

    time_records, psnr_list, ssim_list, lpips_list, fcl_list = [], [], [], [], []
    psnr_ir_list, ssim_ir_list, lpips_ir_list, fcl_ir_list = [], [], [], []
    ndvi_mse_list = []  # 统计所有样本的 NDVI MSE
    sam_rgb_list, sam_ir_list, sam_full_list = [], [], []


    for batch in tqdm(dataloader):
        x_src = batch["conditioning_pixel_values"][:, [2, 1, 0], :, :].cuda()  # ----------------- 选B2, B3, B4
        x_tgt = batch["output_pixel_values"][:, [2, 1, 0], :, :].cuda()        # ----------------- 选B2, B3, B4
        bname = batch["base_name"][0]
        validation_prompt = ""

        with torch.no_grad():
            inference_time, x_pred, x_pred_ir, x_ir = model(args.default, x_src, prompt=validation_prompt, batch=batch)
        time_records.append(inference_time)

        #
        x_pred_rgb = x_pred * 0.5 + 0.5
        x_pred_rgb = x_pred_rgb.clamp(0, 1)

        output_pil = transforms.ToPILImage()(x_pred_rgb[0].cpu())
        input_rgb = x_src * 0.5 + 0.5
        input_rgb = input_rgb.clamp(0, 1)
        input_pil = transforms.ToPILImage()(input_rgb[0].cpu())

        if args.align_method == 'adain':
            output_pil = adain_color_fix(output_pil, input_pil)
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(output_pil, input_pil)

        save_path = os.path.join(args.output_dir, os.path.splitext(bname)[0] + ".png")
        save_rgb_composite(x_pred[0].cpu(), save_path, apply_brightness_enhance=True)


        # === Evaluation ===
        gt_rgb = x_tgt[0] * 0.5 + 0.5
        gt_rgb = gt_rgb.clamp(0, 1)
        gt_pil = transforms.ToPILImage()(gt_rgb.cpu())

        gt_np = np.array(gt_pil).astype(np.float32) / 255.0
        out_np = np.array(output_pil).astype(np.float32) / 255.0

        psnr = peak_signal_noise_ratio(gt_np, out_np, data_range=1.0)
        ssim = structural_similarity(gt_np, out_np, channel_axis=-1, data_range=1.0)

        gt_tensor = F.to_tensor(gt_pil).unsqueeze(0).cuda() * 2 - 1
        out_tensor = F.to_tensor(output_pil).unsqueeze(0).cuda() * 2 - 1
        lpips_score = lpips_loss(gt_tensor, out_tensor).item()

        sr_feat = torch.nn.functional.interpolate(out_tensor, size=(224, 224), mode='bilinear')
        gt_feat = torch.nn.functional.interpolate(gt_tensor, size=(224, 224), mode='bilinear')
        fcl = torch.nn.functional.mse_loss(feature_extractor(sr_feat), feature_extractor(gt_feat)).item()

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips_score)
        fcl_list.append(fcl)

        print(f"{bname} | [RGB] PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpips_score:.4f}, FCL={fcl:.6f}")

        # === IR Evaluation ===

        x_pred_ir_vis = x_pred_ir[0] * 0.5 + 0.5
        x_pred_ir_vis = x_pred_ir_vis.clamp(0, 1)
        ir_pred_pil = transforms.ToPILImage()(x_pred_ir_vis.cpu())

        x_ir_vis = x_ir[0] * 0.5 + 0.5
        x_ir_vis = x_ir_vis.clamp(0, 1)
        ir_gt_pil = transforms.ToPILImage()(x_ir_vis.cpu())

        
        # Save IR image
        ir_save_path = os.path.join(args.output_dir, os.path.splitext(bname)[0] + "_ir.png")
        save_nir_swir_composite(x_pred_ir[0].cpu(), ir_save_path, apply_brightness_enhance=True)

        gt_np_ir = np.array(ir_gt_pil).astype(np.float32) / 255.0
        out_np_ir = np.array(ir_pred_pil).astype(np.float32) / 255.0

        psnr_ir = peak_signal_noise_ratio(gt_np_ir, out_np_ir, data_range=1.0)
        ssim_ir = structural_similarity(gt_np_ir, out_np_ir, channel_axis=-1, data_range=1.0)

        gt_tensor_ir = F.to_tensor(ir_gt_pil).unsqueeze(0).cuda() * 2 - 1
        out_tensor_ir = F.to_tensor(ir_pred_pil).unsqueeze(0).cuda() * 2 - 1
        lpips_ir = lpips_loss(gt_tensor_ir, out_tensor_ir).item()

        sr_feat_ir = torch.nn.functional.interpolate(out_tensor_ir, size=(224, 224), mode='bilinear')
        gt_feat_ir = torch.nn.functional.interpolate(gt_tensor_ir, size=(224, 224), mode='bilinear')
        fcl_ir = torch.nn.functional.mse_loss(feature_extractor(sr_feat_ir), feature_extractor(gt_feat_ir)).item()

        psnr_ir_list.append(psnr_ir)
        ssim_ir_list.append(ssim_ir)
        lpips_ir_list.append(lpips_ir)
        fcl_ir_list.append(fcl_ir)

        print(f"{bname} | [IR] PSNR={psnr_ir:.2f}, SSIM={ssim_ir:.4f}, LPIPS={lpips_ir:.4f}, FCL={fcl_ir:.6f}")

        # === NDVI Evaluation ===
        # 合并 RGB (x_pred) 和 IR (x_pred_ir) 预测波段
        x_pred_full = torch.cat([x_pred[0], x_pred_ir[0]], dim=0)  # (6, H, W)

        # 计算 Ground Truth 和 Prediction 的 NDVI
        ndvi_gt = compute_ndvi(batch["output_pixel_values"][0].cpu())
        ndvi_pred = compute_ndvi(x_pred_full.cpu())

        # 计算 NDVI MSE
        ndvi_mse = torch.mean((ndvi_gt - ndvi_pred) ** 2).item()
        ndvi_mse_list.append(ndvi_mse)

        # 保存 NDVI map
        ndvi_save_path = os.path.join(args.output_dir, os.path.splitext(bname)[0] + "_ndvi.png")
        save_ndvi_map(x_pred_full, ndvi_save_path)

        print(f"{bname} | [NDVI] MSE={ndvi_mse:.6f}")

        # === SAM Evaluation ===
        gt_full_01 = (batch["output_pixel_values"][0].unsqueeze(0).cuda() * 0.5 + 0.5).clamp(0, 1)  # (1,6,H,W)
        gt_rgb_01  = gt_full_01[:, 0:3, :, :]   # [B2,B3,B4]
        gt_ir_01   = gt_full_01[:, 3:6, :, :]   # [B8,B11,B12]

        # Pred
        pred_rgb_01 = (x_pred[0].detach().clamp(-1, 1) * 0.5 + 0.5).unsqueeze(0).cuda()[:, [2,1,0], :, :]
        pred_ir_01  = (x_pred_ir[0].detach().clamp(-1,1) * 0.5 + 0.5).unsqueeze(0).cuda()

        # Full 6-band = [B2,B3,B4,B8,B11,B12]
        pred_full_01 = torch.cat([pred_rgb_01, pred_ir_01], dim=1)

        # Compute SAM (degrees; lower is better)
        sam_rgb  = calculate_sam(gt_rgb_01,  pred_rgb_01)   # 3 bands
        sam_ir   = calculate_sam(gt_ir_01,   pred_ir_01)    # 3 bands
        sam_full = calculate_sam(gt_full_01, pred_full_01)  # 6 bands

        sam_rgb_list.append(float(sam_rgb))
        sam_ir_list.append(float(sam_ir))
        sam_full_list.append(float(sam_full))
        print(f"{bname} | [SAM] RGB={float(sam_rgb):.4f}°, IR={float(sam_ir):.4f}°, Full6={float(sam_full):.4f}°")





    # ========== Summary ==========
    def summarize(metric, name):
        if metric:
            print(f"Avg {name}: {np.mean(metric):.4f}")

    if len(time_records) > 3:
        time_records = time_records[3:]
    summarize(time_records, "Time")
    summarize(psnr_list, "PSNR-RGB")
    summarize(ssim_list, "SSIM-RGB")
    summarize(lpips_list, "LPIPS-RGB")
    summarize(fcl_list, "FCL-RGB")
    summarize(psnr_ir_list, "PSNR-IR")
    summarize(ssim_ir_list, "SSIM-IR")
    summarize(lpips_ir_list, "LPIPS-IR")
    summarize(fcl_ir_list, "FCL-IR")
    summarize(ndvi_mse_list, "NDVI MSE")
    summarize(sam_rgb_list,  "SAM-RGB (deg)")
    summarize(sam_ir_list,   "SAM-IR (deg)")
    summarize(sam_full_list, "SAM-Full6 (deg)")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    # parser.add_argument('--gt_dir', type=str, default='preset/ground_truth', help="path to the ground truth HR images")
    parser.add_argument('--dataset_txt_paths', type=str, default='preset/gt_path_tif_test.txt', help="the directory to LR")
    parser.add_argument('--highquality_dataset_txt_paths', type=str, default='preset/gt_selected_path_tif_test.txt', help="the directory to HR")
    # aux data options
    parser.add_argument("--dem_dataset_txt_paths", type=str, default="preset/dem_tif.txt")
    parser.add_argument("--landcover_dataset_txt_paths", type=str, default="preset/landcover_tif.txt")
    parser.add_argument("--s1_dataset_txt_paths", type=str, default="preset/s1_tif.txt")

    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='preset/models/pisa_sr.pkl', help="path to a model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--resolution_tgt", type=int, default=192)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?") 
    parser.add_argument("--neg_prompt_csd", type=str, default="", help="optional negative prompt string")


    args = parser.parse_args()

    # Call the processing function
    pisa_sr(args)
