import os
import sys
import time
import random
import copy
from types import SimpleNamespace
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook

from SARFusion import SARGuidedOpticalHead


import glob
def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None


import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def initialize_unet(rank_pix, rank_sem, return_lora_module_names=False, pretrained_model_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix = [], [], []
    l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder_pix.append(n.replace(".weight",""))
                l_target_modules_encoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder_pix.append(n.replace(".weight",""))
                l_target_modules_decoder_sem.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others_pix.append(n.replace(".weight",""))
                l_modules_others_sem.append(n.replace(".weight",""))
                break

    lora_conf_encoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_pix)
    lora_conf_decoder_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_pix)
    lora_conf_others_pix = LoraConfig(r=rank_pix, init_lora_weights="gaussian",target_modules=l_modules_others_pix)
    lora_conf_encoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_encoder_sem)
    lora_conf_decoder_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_target_modules_decoder_sem)
    lora_conf_others_sem = LoraConfig(r=rank_sem, init_lora_weights="gaussian",target_modules=l_modules_others_sem)

    unet.add_adapter(lora_conf_encoder_pix, adapter_name="default_encoder_pix")
    unet.add_adapter(lora_conf_decoder_pix, adapter_name="default_decoder_pix")
    unet.add_adapter(lora_conf_others_pix, adapter_name="default_others_pix")
    unet.add_adapter(lora_conf_encoder_sem, adapter_name="default_encoder_sem")
    unet.add_adapter(lora_conf_decoder_sem, adapter_name="default_decoder_sem")
    unet.add_adapter(lora_conf_others_sem, adapter_name="default_others_sem")

    if return_lora_module_names:
        return unet, l_target_modules_encoder_pix, l_target_modules_decoder_pix, l_modules_others_pix, l_target_modules_encoder_sem, l_target_modules_decoder_sem, l_modules_others_sem
    else:
        return unet


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()


class PostVAEFusion(nn.Module):
    def __init__(self, s1_in_ch=1, rgb_ch=3, mid_ch=32):
        super().__init__()
        in_ch = rgb_ch + s1_in_ch
        self.in_proj = nn.Conv2d(in_ch, mid_ch, kernel_size=1)
        self.dw = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch)
        self.pw = nn.Conv2d(mid_ch, mid_ch, kernel_size=1)
        self.d1 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=3, dilation=3)
        self.out_proj = nn.Conv2d(mid_ch * 3, rgb_ch, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.act = nn.GELU()

    def forward(self, rgb_out, s1_img):
        if s1_img.shape[-2:] != rgb_out.shape[-2:]:
            s1_img = F.interpolate(s1_img, size=rgb_out.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([rgb_out, s1_img], dim=1)
        x = self.in_proj(x)
        x = self.act(self.pw(self.dw(x)))
        b1 = self.act(self.d1(x))
        b2 = self.act(self.d2(x))
        b3 = self.act(self.d3(x))
        x = torch.cat([b1, b2, b3], dim=1)
        delta = self.out_proj(x)
        return (rgb_out + self.gamma * delta).clamp(-1, 1)

class LSSR(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
        self.args = args
        self.device = "cuda"


        if args.resume_ckpt is None:
            self.unet, lora_unet_modules_encoder_pix, lora_unet_modules_decoder_pix, lora_unet_others_pix, \
                lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem, =\
                    initialize_unet(rank_pix=args.lora_rank_unet_pix, rank_sem=args.lora_rank_unet_sem, pretrained_model_path=args.pretrained_model_path, return_lora_module_names=True)
            
            self.lora_rank_unet_pix = args.lora_rank_unet_pix
            self.lora_rank_unet_sem = args.lora_rank_unet_sem
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
                self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
                lora_unet_modules_encoder_pix, lora_unet_modules_decoder_pix, lora_unet_others_pix, \
                    lora_unet_modules_encoder_sem, lora_unet_modules_decoder_sem, lora_unet_others_sem
        else:
            print(f'====> resume from {args.resume_ckpt}')
            stage1_yaml = find_filepath(args.resume_ckpt.split('/checkpoints')[0], 'hparams.yml')
            stage1_args = read_yaml(stage1_yaml)
            stage1_args = SimpleNamespace(**stage1_args)
            self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")
            self.lora_rank_unet_pix = stage1_args.lora_rank_unet_pix
            self.lora_rank_unet_sem = stage1_args.lora_rank_unet_sem
            pisasr = torch.load(args.resume_ckpt)
            self.load_ckpt_from_state_dict(pisasr)
        # unet.enable_xformers_memory_efficient_attention()
        self.unet.to("cuda")
        self.vae_fix = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.vae_fix.to('cuda')

        self.timesteps1 = torch.tensor([args.timesteps1], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae_fix.requires_grad_(False)
        self.vae_fix.eval()

        # aux data embedding
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 192→96
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 96→48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48→24
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 输出通道和 latent 匹配（4）
        )

        self.lc_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 192→96
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 96→48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48→24
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 输出通道和 latent 匹配（4）
        )

        self.month_encoder = nn.Embedding(5, 4)  # 5个月，每月映射到4 channels

        # cross attention
        self.latent_project_in = nn.Conv2d(4, 256, kernel_size=1)
        self.latent_project_out = nn.Conv2d(256, 4, kernel_size=1)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.gamma_rgb = nn.Parameter(torch.tensor(1.0))
        self.gamma_ir = nn.Parameter(torch.tensor(1.0))

        # post VAE optical-SAR fusion
        # self.post_fusion = PostVAEFusion(s1_in_ch=2, rgb_ch=3, mid_ch=64)
        self.post_fusion = SARGuidedOpticalHead(opt_in_ch=3, sar_in_ch=2, dim=96, heads=6, layers=2,
                                 mlp_ratio=4.0, window_size=12, gating=True, out_act=None)


    def set_train_pix(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "pix" in n:
                _p.requires_grad = True
            if "sem" in n:
                _p.requires_grad = False
    
    def set_train_sem(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "sem" in n:
                _p.requires_grad = True
            if "pix" in n:
                _p.requires_grad = False

    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")

        self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix, \
        self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem= \
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"], \
            sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"]

        for n, p in self.unet.named_parameters():
            if "lora" in n:
                p.data.copy_(sd["state_dict_unet"][n])

    # Adopted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def forward(self, c_t, c_tgt, batch=None, args=None):

        bs = c_t.shape[0]
        # print("🟡 [Input] input image shape:", c_t.shape)   # torch.Size([1, 3, 192, 192])

        dem = batch["dem_values"]                      # (1, 1, 192, 192)
        lc = batch["landcover_values"].float()         # (1, 1, 192, 192)
        # print("🟡 [Input] dem shape:", dem.shape)
        # print("🟡 [Input] lc shape:", lc.shape)

        ir_lr = batch["ir_lr_values"]
        ir_hr = batch["ir_hr_values"]
        # print("🟡 [Input] ir_lr_values shape:", ir_lr.shape)    # torch.Size([1, 3, 192, 192])
        # print("🟡 [Input] ir_hr_values shape:", ir_hr.shape)    # torch.Size([1, 3, 192, 192])
        
        # -------- rgb embedding -------- 
        encoded_control = self.vae_fix.encode(c_t).latent_dist.sample() * self.vae_fix.config.scaling_factor
        # print("🔵 [VAE] encoded_control shape:", encoded_control.shape) # torch.Size([1, 4, 24, 24])

        # -------- ir embedding -------- 
        dtype = encoded_control.dtype
        device = encoded_control.device
        ir_lr = ir_lr.to(dtype=self.vae_fix.dtype, device=self.device)
        # ir_emb = self.ir2rgb(ir_lr.to(dtype=dtype, device=device))
        encoded_control_ir = self.vae_fix.encode(ir_lr).latent_dist.sample() * self.vae_fix.config.scaling_factor
        # print("🔵 [VAE] encoded_control_ir shape:", encoded_control_ir.shape) # torch.Size([1, 4, 24, 24])

        # ------- aux data injection ---------------
        dem_feat = self.dem_encoder(dem.to(dtype=dtype, device=device))
        lc_feat = self.lc_encoder(lc.to(dtype=dtype, device=device))
        # print("🟡 [Input] dem feature shape:", dem_feat.shape)      # torch.Size([1, 4, 24, 24])
        # print("🟡 [Input] landcover feature shape:", lc_feat.shape) # torch.Size([1, 4, 24, 24])

        # print(batch["month_index"].item())
        month_feat = self.month_encoder(batch["month_index"].to(device=device))  # (B, 4)
        month_feat = month_feat.view(bs, 4, 1, 1)                # (B, 4, 1, 1)
        # print("🟡 [Input] time feature shape:", month_feat.shape)

        # 1. 构建 knowledge_feat
        knowledge_feat = dem_feat + lc_feat + month_feat  # [B, 4, 24, 24]

        # 2. 将 latent + knowledge 投入 cross-attention
        rgb_proj = self.latent_project_in(encoded_control)        # [B, 256, 24, 24]
        ir_proj = self.latent_project_in(encoded_control_ir)        # [B, 256, 24, 24]
        knowledge_proj = self.latent_project_in(knowledge_feat) # [B, 256, 24, 24]

        rgb_proj_flat = rgb_proj.flatten(2).transpose(1, 2)          # [B, 576, 256]
        ir_proj_flat = ir_proj.flatten(2).transpose(1, 2)          # [B, 576, 256]
        knowledge_flat = knowledge_proj.flatten(2).transpose(1, 2)  # [B, 576, 256]

        # 3. Cross-Attn: query = latent, key/value = knowledge
        rgb_attn_output, _ = self.cross_attn(
            query=rgb_proj_flat,
            key=knowledge_flat,
            value=knowledge_flat
        )  # [B, 576, 256]

        ir_attn_output, _ = self.cross_attn(
            query=ir_proj_flat,
            key=knowledge_flat,
            value=knowledge_flat
        )  # [B, 576, 256]

        # 4. 反变换成 [B, 4, 24, 24]
        rgb_attn_output = rgb_attn_output.transpose(1, 2).view(bs, 256, 24, 24)
        rgb_attn_output = self.latent_project_out(rgb_attn_output)  # [B, 4, 24, 24]

        ir_attn_output = ir_attn_output.transpose(1, 2).view(bs, 256, 24, 24)
        ir_attn_output = self.latent_project_out(ir_attn_output)  # [B, 4, 24, 24]

        # 5. 注入 latent（残差连接）
        encoded_control = encoded_control + self.gamma_rgb * rgb_attn_output
        encoded_control_ir = encoded_control_ir + self.gamma_ir * ir_attn_output

        # ---------- prompt_embeddings and neg_prompt_embeddings -----------
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])
        null_prompt_embeds = self.encode_prompt(batch["null_prompt"])
        # print("🟢 [Prompt] prompt_embeds shape:", prompt_embeds.shape)  # torch.Size([1, 77, 1024])
        # print("🟢 [Prompt] neg_prompt_embeds shape:", neg_prompt_embeds.shape)  # torch.Size([1, 77, 1024])

        if random.random() < args.null_text_ratio:
            pos_caption_enc = null_prompt_embeds
            # print("🟣 Using NULL text embedding.") 
        else:
            pos_caption_enc = prompt_embeds
            # print("🟣 Using regular prompt embedding.")

        # ------------rgb pred-----------------
        model_pred = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=pos_caption_enc.to(torch.float32),).sample
        # print("🔴 [UNet] model_pred shape:", model_pred.shape)  # torch.Size([1, 4, 24, 24])
        # ------------ir pred-----------------
        model_pred_ir = self.unet(encoded_control_ir, self.timesteps1, encoder_hidden_states=pos_caption_enc.to(torch.float32),).sample
        # print("🔴 [UNet] model_pred_ir shape:", model_pred_ir.shape)  # torch.Size([1, 4, 24, 24])

        x_denoised = encoded_control - model_pred
        # print("🟠 [Latent] x_denoised shape:", x_denoised.shape)    # torch.Size([1, 4, 24, 24])
        x_denoised_ir = encoded_control_ir - model_pred_ir
        # print("🟠 [Latent] x_denoised_ir shape:", x_denoised_ir.shape) 

        output_image = (self.vae_fix.decode(x_denoised / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)
        # print("🟣 [Output] output_image shape:", output_image.shape)    # torch.Size([1, 3, 192, 192])
        output_image_ir = (self.vae_fix.decode(x_denoised_ir / self.vae_fix.config.scaling_factor).sample).clamp(-1, 1)
        # print("🟣 [Output] output_image_ir shape:", output_image_ir.shape)    # torch.Size([1, 3, 192, 192])

        # ------------fusion with sentinel1-----------------
        s1_hr = batch["s1_values"]
        output_image = self.post_fusion(output_image.to(dtype=dtype, device=device), s1_hr.to(dtype=dtype, device=device))
        output_image_ir = self.post_fusion(output_image_ir.to(dtype=dtype, device=device), s1_hr.to(dtype=dtype, device=device))

        return output_image, x_denoised, output_image_ir, ir_hr, prompt_embeds, neg_prompt_embeds


    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules_pix"], sd["unet_lora_decoder_modules_pix"], sd["unet_lora_others_modules_pix"] =\
            self.lora_unet_modules_encoder_pix, self.lora_unet_modules_decoder_pix, self.lora_unet_others_pix
        sd["unet_lora_encoder_modules_sem"], sd["unet_lora_decoder_modules_sem"], sd["unet_lora_others_modules_sem"] =\
            self.lora_unet_modules_encoder_sem, self.lora_unet_modules_decoder_sem, self.lora_unet_others_sem
        sd["lora_rank_unet_pix"] = self.lora_rank_unet_pix
        sd["lora_rank_unet_sem"] = self.lora_rank_unet_sem
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}

        # ✅ 保存 aux encoder 权重
        # print("saving aux encoders...")
        sd["dem_encoder"] = self.dem_encoder.state_dict()
        sd["lc_encoder"]  = self.lc_encoder.state_dict()
        sd["month_encoder"] = self.month_encoder.state_dict()
        sd["latent_project_in"] = self.latent_project_in.state_dict()
        sd["latent_project_out"] = self.latent_project_out.state_dict()
        sd["cross_attn"] = self.cross_attn.state_dict()
        sd["gamma_rgb"] = self.gamma_rgb.detach().cpu()
        sd["gamma_ir"] = self.gamma_ir.detach().cpu()
        sd["post_fusion"] = self.post_fusion.state_dict()

        torch.save(sd, outf)


class LSSR_eval(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = "cuda"
        self.weight_dtype = self._get_dtype(args.mixed_precision)
        self.args = args

        # Initialize components
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").to(self.device)
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet")

        # aux data embedding
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 192→96
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 96→48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48→24
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 输出通道和 latent 匹配（4）
        ).to(device=self.device, dtype=self.weight_dtype)

        self.lc_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 192→96
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 96→48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48→24
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 输出通道和 latent 匹配（4）
        ).to(device=self.device, dtype=self.weight_dtype)

        self.month_encoder = nn.Embedding(5, 4).to(device=self.device)  # 5个月，每月映射到4 channels

        # cross attention
        self.latent_project_in = nn.Conv2d(4, 256, kernel_size=1).to(device=self.device, dtype=self.weight_dtype)
        self.latent_project_out = nn.Conv2d(256, 4, kernel_size=1).to(device=self.device, dtype=self.weight_dtype)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True).to(device=self.device, dtype=self.weight_dtype)
        self.gamma_rgb = nn.Parameter(torch.tensor(1.0)).to(device=self.device, dtype=self.weight_dtype)
        self.gamma_ir = nn.Parameter(torch.tensor(1.0)).to(device=self.device, dtype=self.weight_dtype)

        # post VAE optical-SAR fusion
        # self.post_fusion = PostVAEFusion(s1_in_ch=2, rgb_ch=3, mid_ch=64).to(device=self.device, dtype=self.weight_dtype)
        self.post_fusion = SARGuidedOpticalHead(opt_in_ch=3, sar_in_ch=2, dim=96, heads=6, layers=2,
                                 mlp_ratio=4.0, window_size=12, gating=True, out_act=None).to(device=self.device, dtype=self.weight_dtype)

        # Load pretrained weights
        self._load_pretrained_weights(args.pretrained_path)

        # Initialize VAE tiling
        self._init_tiled_vae(
            encoder_tile_size=args.vae_encoder_tiled_size,
            decoder_tile_size=args.vae_decoder_tiled_size
        )

        # Prepare LoRA adapters
        if not args.default:
            self._prepare_lora_deltas(["default_encoder_sem", "default_decoder_sem", "default_others_sem"])
        set_weights_and_activate_adapters(self.unet, ["default_encoder_sem", "default_decoder_sem", "default_others_sem"], [1.0, 1.0, 1.0])

        # lora_total = sum(p.numel() for n, p in self.unet.named_parameters() if "lora" in n) / 1e6
        # lora_trainable = sum(p.numel() for n, p in self.unet.named_parameters() if "lora" in n and p.requires_grad) / 1e6
        # print("\n==== LoRA Adapters Parameter Count (in Millions) ====")
        # print(f"{'LoRA Adapters':20s} | Total: {lora_total:.3f}M | Trainable: {lora_trainable:.3f}M")

        self.unet.merge_and_unload()

        # Move models to device and precision
        self._move_models_to_device_and_dtype()

        # Set parameters
        self.timesteps1 = torch.tensor([1], device=self.device).long()
        self.lambda_pix = torch.tensor([args.lambda_pix], device=self.device)
        self.lambda_sem = torch.tensor([args.lambda_sem], device=self.device)


    def _get_dtype(self, precision):
        """Get the appropriate data type based on precision."""
        if precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _move_models_to_device_and_dtype(self):
        """Move models to the correct device and precision."""
        for model in [self.vae, self.unet, self.text_encoder]:
            model.to(self.device, dtype=self.weight_dtype)
            model.requires_grad_(False)

    def _load_pretrained_weights(self, pretrained_path):
        """Load pretrained weights and initialize LoRA adapters."""
        sd = torch.load(pretrained_path)
        self._load_and_save_ckpt_from_state_dict(sd)

    def _prepare_lora_deltas(self, adapter_names):
        """Precompute and store LoRA deltas for the given adapters."""
        self.lora_deltas_sem = {}
        key_list = [key for key, _ in self.unet.named_modules() if "lora_" not in key]

        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.unet, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    for active_adapter in adapter_names:
                        if active_adapter in target.lora_A.keys():
                            base_layer = target.get_base_layer()
                            weight_A = target.lora_A[active_adapter].weight
                            weight_B = target.lora_B[active_adapter].weight

                            s = target.get_base_layer().weight.size()
                            if s[2:4] == (1, 1):  # Conv2D 1x1
                                output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * target.scaling[active_adapter]
                            elif len(s) == 2:  # Linear layer
                                output_tensor = transpose(weight_B @ weight_A, False) * target.scaling[active_adapter]
                            else:  # Conv2D 3x3
                                output_tensor = F.conv2d(
                                    weight_A.permute(1, 0, 2, 3),
                                    weight_B,
                                ).permute(1, 0, 2, 3) * target.scaling[active_adapter]

                            key = key + ".weight"
                            self.lora_deltas_sem[key] = output_tensor.data.to(dtype=self.weight_dtype, device=self.device)

    def _apply_lora_delta(self):
        """Merge LoRA deltas into UNet weights."""
        for name, param in self.unet.named_parameters():
            if name in self.lora_deltas_sem:
                param.data = self.lora_deltas_sem[name] + self.ori_unet_weight[name]
            else:
                param.data = self.ori_unet_weight[name]

    def _apply_ori_weight(self):
        """Restore original UNet weights."""
        for name, param in self.unet.named_parameters():
            param.data = self.ori_unet_weight[name]

    def _load_and_save_ckpt_from_state_dict(self, sd):
        """Load checkpoint and initialize LoRA adapters."""
        # Define LoRA configurations
        self.lora_conf_encoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_pix"])
        self.lora_conf_decoder_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_pix"])
        self.lora_conf_others_pix = LoraConfig(r=sd["lora_rank_unet_pix"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_pix"])

        self.lora_conf_encoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules_sem"])
        self.lora_conf_decoder_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules_sem"])
        self.lora_conf_others_sem = LoraConfig(r=sd["lora_rank_unet_sem"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules_sem"])

        # Add and load adapters
        self.unet.add_adapter(self.lora_conf_encoder_pix, adapter_name="default_encoder_pix")
        self.unet.add_adapter(self.lora_conf_decoder_pix, adapter_name="default_decoder_pix")
        self.unet.add_adapter(self.lora_conf_others_pix, adapter_name="default_others_pix")

        for name, param in self.unet.named_parameters():
            if "pix" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        # Merge and save unet weights
        set_weights_and_activate_adapters(self.unet, ["default_encoder_pix", "default_decoder_pix", "default_others_pix"], [1.0, 1.0, 1.0])
        self.unet.merge_and_unload()
        self.ori_unet_weight = {}
        for name, param in self.unet.named_parameters():
            self.ori_unet_weight[name] = param.clone()
            self.ori_unet_weight[name] = self.ori_unet_weight[name].data.to(self.weight_dtype).to("cuda")
        
        # Add semantic adapters
        self.unet.add_adapter(self.lora_conf_encoder_sem, adapter_name="default_encoder_sem")
        self.unet.add_adapter(self.lora_conf_decoder_sem, adapter_name="default_decoder_sem")
        self.unet.add_adapter(self.lora_conf_others_sem, adapter_name="default_others_sem")
        
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.data.copy_(sd["state_dict_unet"][name])

        # aux encoder loading
        if "dem_encoder" in sd:
            print("loading dem encoder...")
            self.dem_encoder.load_state_dict(sd["dem_encoder"])
        if "lc_encoder" in sd:
            print("loading landcover encoder...")
            self.lc_encoder.load_state_dict(sd["lc_encoder"])
        if "month_encoder" in sd:
            print("loading month_encoder encoder...")
            self.month_encoder.load_state_dict(sd["month_encoder"])
        
        if "latent_project_in" in sd:
            print("loading latent_project_in encoder...")
            self.latent_project_in.load_state_dict(sd["latent_project_in"])
        if "latent_project_out" in sd:
            print("loading latent_project_out encoder...")
            self.latent_project_out.load_state_dict(sd["latent_project_out"])
        if "cross_attn" in sd:
            print("loading cross_attn encoder...")
            self.cross_attn.load_state_dict(sd["cross_attn"])
        if "gamma_rgb" in sd:
            print("loading gamma_rgb encoder...")
            self.gamma_rgb = nn.Parameter(torch.tensor(sd["gamma_rgb"]))
        if "gamma_ir" in sd:
            print("loading gamma_ir encoder...")
            self.gamma_ir = nn.Parameter(torch.tensor(sd["gamma_ir"]))

        if "post_fusion" in sd:
            print("loading post_fusion encoder...")
            self.post_fusion.load_state_dict(sd["post_fusion"])

    def set_eval(self):
        """Set models to evaluation mode."""
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def encode_prompt(self, prompt_batch):
        """Encode text prompts into embeddings."""
        with torch.no_grad():
            prompt_embeds = [
                self.text_encoder(
                    self.tokenizer(
                        caption, max_length=self.tokenizer.model_max_length,
                        padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(self.text_encoder.device)
                )[0]
                for caption in prompt_batch
            ]
        return torch.concat(prompt_embeds, dim=0)

    def count_parameters(self, model):
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters()) / 1e9

    @torch.no_grad()
    def forward(self, default, c_t, prompt=None, batch=None):
        """Forward pass for inference."""
        torch.cuda.synchronize()
        start_time = time.time()

        c_t = c_t.to(dtype=self.weight_dtype)
        prompt_embeds = self.encode_prompt([prompt]).to(dtype=self.weight_dtype)

        bs = c_t.shape[0]

        dem = batch["dem_values"]                      # (1, 1, 192, 192)
        lc = batch["landcover_values"].float()         # (1, 1, 192, 192)
        # print("🟡 [Input] dem shape:", dem.shape)
        # print("🟡 [Input] lc shape:", lc.shape)

        ir_lr = batch["ir_lr_values"]
        ir_hr = batch["ir_hr_values"]
        # print("🟡 [Input] ir_lr_values shape:", ir_lr.shape)    # torch.Size([1, 3, 192, 192])
        # print("🟡 [Input] ir_hr_values shape:", ir_hr.shape)    # torch.Size([1, 3, 192, 192])
        
        # -------- rgb embedding -------- 
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        # print("🔵 [VAE] encoded_control shape:", encoded_control.shape) # torch.Size([1, 4, 24, 24])
        
        # -------- ir embedding -------- 
        # ir_emb = self.ir2rgb(ir_lr)
        ir_lr = ir_lr.to(self.device, dtype=self.weight_dtype)
        encoded_control_ir = self.vae.encode(ir_lr).latent_dist.sample() * self.vae.config.scaling_factor
        # print("🔵 [VAE] encoded_control_ir shape:", encoded_control_ir.shape) # torch.Size([1, 4, 24, 24])


        # ------- aux data injection -----------
        dtype = encoded_control.dtype
        device = encoded_control.device

        dem_feat = self.dem_encoder(dem.to(dtype=dtype, device=device))
        lc_feat = self.lc_encoder(lc.to(dtype=dtype, device=device))

        # print(batch["month_index"])
        month_feat = self.month_encoder(batch["month_index"].to(device=device)).to(dtype=encoded_control.dtype)  # (B, 4)
        month_feat = month_feat.view(bs, 4, 1, 1)                # (B, 4, 1, 1)
        # print("🟡 [Input] time feature shape:", month_feat.shape)

        # 1. 构建 knowledge_feat
        knowledge_feat = dem_feat + lc_feat + month_feat  # [B, 4, 24, 24]

        # 2. 将 latent + knowledge 投入 cross-attention
        rgb_proj = self.latent_project_in(encoded_control)        # [B, 256, 24, 24]
        ir_proj = self.latent_project_in(encoded_control_ir)        # [B, 256, 24, 24]
        knowledge_proj = self.latent_project_in(knowledge_feat) # [B, 256, 24, 24]

        rgb_proj_flat = rgb_proj.flatten(2).transpose(1, 2)          # [B, 576, 256]
        ir_proj_flat = ir_proj.flatten(2).transpose(1, 2)          # [B, 576, 256]
        knowledge_flat = knowledge_proj.flatten(2).transpose(1, 2)  # [B, 576, 256]

        # 3. Cross-Attn: query = latent, key/value = knowledge
        rgb_attn_output, _ = self.cross_attn(
            query=rgb_proj_flat,
            key=knowledge_flat,
            value=knowledge_flat
        )  # [B, 576, 256]

        ir_attn_output, _ = self.cross_attn(
            query=ir_proj_flat,
            key=knowledge_flat,
            value=knowledge_flat
        )  # [B, 576, 256]

        # 4. 反变换成 [B, 4, 24, 24]
        rgb_attn_output = rgb_attn_output.transpose(1, 2).view(bs, 256, 24, 24)
        rgb_attn_output = self.latent_project_out(rgb_attn_output)  # [B, 4, 24, 24]

        ir_attn_output = ir_attn_output.transpose(1, 2).view(bs, 256, 24, 24)
        ir_attn_output = self.latent_project_out(ir_attn_output)  # [B, 4, 24, 24]

        # 5. 注入 latent（残差连接）
        encoded_control = encoded_control + self.gamma_rgb * rgb_attn_output
        encoded_control_ir = encoded_control_ir + self.gamma_ir * ir_attn_output
        
        # Tile and process latents if necessary
        model_pred = self._process_latents(encoded_control, prompt_embeds, default)
        model_pred_ir = self._process_latents(encoded_control_ir, prompt_embeds, default)

        # Decode output
        x_denoised = encoded_control - model_pred
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        x_denoised_ir = encoded_control_ir - model_pred_ir
        output_image_ir = self.vae.decode(x_denoised_ir / self.vae.config.scaling_factor).sample.clamp(-1, 1)

        # ------------fusion with sentinel1-----------------
        # print("fusion with sentinel1 ...")
        s1_hr = batch["s1_values"]
        output_image = self.post_fusion(output_image.to(dtype=dtype, device=device), s1_hr.to(dtype=dtype, device=device))
        output_image_ir = self.post_fusion(output_image_ir.to(dtype=dtype, device=device), s1_hr.to(dtype=dtype, device=device))
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        return total_time, output_image, output_image_ir, ir_hr

    def _process_latents(self, encoded_control, prompt_embeds, default):
        """Process latents with or without tiling."""
        h, w = encoded_control.size()[-2:]
        tile_size, tile_overlap = self.args.latent_tiled_size, self.args.latent_tiled_overlap

        if h * w <= tile_size * tile_size:
            print("[Tiled Latent]: Input size is small, no tiling required.")
            return self._predict_no_tiling(encoded_control, prompt_embeds, default)

        print(f"[Tiled Latent]: Input size {h}x{w}, tiling required.")
        return self._predict_with_tiling(encoded_control, prompt_embeds, default, tile_size, tile_overlap)

    def _predict_no_tiling(self, encoded_control, prompt_embeds, default):
        """Predict on the entire latent without tiling."""
        if default:
            return self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample

        model_pred_sem = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_ori_weight()
        model_pred_pix = self.unet(encoded_control, self.timesteps1, encoder_hidden_states=prompt_embeds).sample
        self._apply_lora_delta()

        model_pred_sem -= model_pred_pix
        return self.lambda_pix * model_pred_pix + self.lambda_sem * model_pred_sem

    def _predict_with_tiling(self, encoded_control, prompt_embeds, default, tile_size, tile_overlap):
        """Predict on the latent with tiling."""
        _, _, h, w = encoded_control.size()
        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
        tile_size = min(tile_size, min(h, w))
        grid_rows = 0
        cur_x = 0
        while cur_x < encoded_control.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < encoded_control.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        noise_preds = []
        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # input tile dimensions
                input_tile = encoded_control[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)

                if len(input_list) == 1 or col == grid_cols-1:
                    input_list_t = torch.cat(input_list, dim=0)
                    # predict the noise residual
                    if default:
                        print(f"[0:Default setting]")
                        model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                    else:
                        print(f"[1:Adjustable setting]")
                        model_out_sem = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                        self._apply_ori_weight()
                        model_out_pix = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds,).sample
                        self._apply_lora_delta()
                        model_out_sem = model_out_sem - model_out_pix
                        model_out = self.lambda_pix * model_out_pix + self.lambda_sem * model_out_sem
                    # model_out = self.unet(input_list_t, self.timesteps1, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
                    input_list = []
                noise_preds.append(model_out)

        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(encoded_control.shape, device=encoded_control.device)
        contributors = torch.zeros(encoded_control.shape, device=encoded_control.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        model_pred = noise_pred
        return model_pred


    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generate a Gaussian mask for tile contributions."""
        from numpy import pi, exp, sqrt
        import numpy as np

        midpoint_x = (tile_width - 1) / 2
        midpoint_y = (tile_height - 1) / 2
        x_probs = [exp(-(x - midpoint_x) ** 2 / (2 * (tile_width ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for x in range(tile_width)]
        y_probs = [exp(-(y - midpoint_y) ** 2 / (2 * (tile_height ** 2) * 0.01)) / sqrt(2 * pi * 0.01) for y in range(tile_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tensor(weights, device=self.device).repeat(nbatches, self.unet.config.in_channels, 1, 1)

    def _init_tiled_vae(self, encoder_tile_size=256, decoder_tile_size=256, fast_decoder=False, fast_encoder=False, color_fix=False, vae_to_gpu=True):
        """Initialize VAE with tiled encoding/decoding."""
        encoder, decoder = self.vae.encoder, self.vae.decoder

        if not hasattr(encoder, 'original_forward'):
            encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'):
            decoder.original_forward = decoder.forward

        encoder.forward = VAEHook(encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        decoder.forward = VAEHook(decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)