import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from lssr import CSDLoss, LSSR
from src.my_utils.training_utils import parse_args  
from src.datasets.dataset_lssr import PairedSROnlineTxtDataset

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix, compute_fft_loss, compute_ndvi_loss, save_rgb_composite, save_nir_swir_composite
import random


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    # initialize LSSR model
    net_lssr = LSSR(args)
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_lssr.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_lssr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init CSDLoss model
    net_csd = CSDLoss(args=args, accelerator=accelerator)
    net_csd.requires_grad_(False)
    # init lpips model
    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # # set gen adapter
    net_lssr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix'])
    net_lssr.set_train_pix() # first to remove degradation

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_lssr.unet.named_parameters():
        if "lora" in n:
            layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    
    # initialize the dataset
    dataset_train = PairedSROnlineTxtDataset(split="train", args=args)
    dataset_val = PairedSROnlineTxtDataset(split="eval", args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    

    # init RAM for text prompt extractor
    from ram.models.ram_lora import ram
    from ram import inference_ram as inference
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='src/ram_pretrain_model/ram_swin_large_14m.pth',
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    net_lssr, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_lssr, optimizer, dl_train, lr_scheduler
    )
    net_lpips = accelerator.prepare(net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    lambda_l2 = args.lambda_l2
    lambda_lpips = 0
    lambda_csd = 0
    lambda_fft = args.lambda_fft
    lambda_ndvi = args.lambda_ndvi

    if args.resume_ckpt is not None:
        args.pix_steps = 1
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            with accelerator.accumulate(net_lssr):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]

                # get text prompts from GT
                # x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                x_tgt_rgb = x_tgt[:, [2,1,0], :, :]  # B2, B3, B4
                x_tgt_ram = ram_transforms(x_tgt_rgb*0.5+0.5)

                caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                batch["prompt"] = [f'{each_caption}, {args.pos_prompt_csd}' for each_caption in caption]
                
                if global_step == args.pix_steps:
                    # begin the semantic optimization
                    net_lssr.unet.set_adapter(['default_encoder_pix', 'default_decoder_pix', 'default_others_pix','default_encoder_sem', 'default_decoder_sem', 'default_others_sem'])
                    net_lssr.set_train_sem()

                    lambda_l2 = args.lambda_l2
                    lambda_lpips = args.lambda_lpips
                    lambda_csd = args.lambda_csd
                    lambda_fft = args.lambda_fft
                    lambda_ndvi = args.lambda_ndvi
                    
                x_tgt = x_tgt[:, [2,1,0], :, :] # B2, B3, B4
                x_src = x_src[:, [2,1,0], :, :] # B2, B3, B4
                # print("x_tgt.shape", x_tgt.shape)
                # print("x_src.shape", x_src.shape)
                x_tgt_pred, latents_pred, x_ir_pred, x_ir_gt, prompt_embeds, neg_prompt_embeds = net_lssr(x_src, x_tgt, batch=batch, args=args)
                # print("x_tgt_pred.shape", x_tgt_pred.shape)

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * lambda_l2 \
                            + F.mse_loss(x_ir_pred.float(), x_ir_gt.float(), reduction="mean") * lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * lambda_lpips \
                            + net_lpips(x_ir_pred.float(), x_ir_gt.float()).mean() * lambda_lpips
                loss_fft = compute_fft_loss(x_tgt_pred.float(), x_tgt.float()) * lambda_fft \
                            + compute_fft_loss(x_ir_pred.float(), x_ir_gt.float()) * lambda_fft

                x_sr_full = torch.cat([x_tgt_pred, x_ir_pred], dim=1)
                x_gt_full = torch.cat([x_tgt, x_ir_gt], dim=1)
                # NDVI requires [0, 1] scale
                x_sr_full = (x_sr_full + 1) / 2
                x_gt_full = (x_gt_full + 1) / 2
                loss_ndvi = compute_ndvi_loss(x_sr_full, x_gt_full, red_index=0, nir_index=3) * lambda_ndvi

                loss = loss_l2 + loss_lpips + loss_fft + loss_ndvi
                # reg loss
                loss_csd = net_csd.cal_csd(latents_pred, prompt_embeds, neg_prompt_embeds, args, ) * lambda_csd
                loss = loss + loss_csd
                if torch.isnan(loss):
                    print("NaN detected at step", global_step)
                    print("loss_csd:", loss_csd.item(), "loss_l2:", loss_l2.item(), "loss_lpips:", loss_lpips.item())
                    raise ValueError("NaN occurred!")


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_csd"] = loss_csd.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss_fft"] = loss_fft.detach().item()
                    logs["loss_ndvi"] = loss_ndvi.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_lssr).save_model(outf)

                    # test
                    if global_step % args.eval_freq == 1:
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()

                            x_tgt = x_tgt[:, [2,1,0], :, :] # B2, B3, B4
                            x_src = x_src[:, [2,1,0], :, :] # B2, B3, B4
                            # print("x_tgt_pred.shape", x_tgt_pred.shape)

                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # get text prompts from LR
                                x_src_ram = ram_transforms(x_src * 0.5 + 0.5)
                                caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                batch_val["prompt"] = caption
                                # forward pass
                                x_tgt_pred, latents_pred, x_ir_pred, x_ir_gt, _, _ = accelerator.unwrap_model(net_lssr)(x_src, x_tgt,
                                                                                                      batch=batch_val,
                                                                                                      args=args)

                                x_basename = os.path.splitext(x_basename)[0]
                                outf_rgb = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}.png")
                                save_rgb_composite(x_tgt_pred[0].detach().cpu(), outf_rgb, apply_brightness_enhance=True)

                                outf_ir = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}_ir.png")
                                save_nir_swir_composite(x_ir_pred[0].detach().cpu(), outf_ir, apply_brightness_enhance=True)


                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)

                    accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    args = parse_args()
    main(args)
