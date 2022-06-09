# Standard Library
from typing import List

import clip
import loguru
import lpips
import torch
import torchvision.transforms as T
from constants import DiffusionModelEnum
from guided_diffusion.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.script_util import create_gaussian_diffusion, create_model
from settings import (
    clip_model_settings,
    diffusion_model_settings,
    generation_settings,
    torch_model_settings,
)


def get_normalize() -> T.Normalize:
    """
    Return Image Normalizer
    Returns:
        T.Normalize: Image Normalizer
    """
    loguru.logger.info("Load Image Normalizer")
    return T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )


def load_clip_model() -> List[clip.model.CLIP]:
    """
    Return List of Clip Models
    """
    clip_models = []
    device = torch_model_settings.device
    if clip_model_settings.ViTB32 is True:
        loguru.logger.info("Load ViTB32")
        clip_models.append(
            clip.load("ViT-B/32", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.ViTB16 is True:
        loguru.logger.info("Load ViTB16")
        clip_models.append(
            clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.ViTL14 is True:
        loguru.logger.info("Load ViTL14")
        clip_models.append(
            clip.load("ViT-L/14", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN50 is True:
        loguru.logger.info("Load RN50")
        clip_models.append(clip.load("RN50", jit=False)[0].eval().requires_grad_(False).to(device))
    if clip_model_settings.RN50x4 is True:
        loguru.logger.info("Load RN50x4")
        clip_models.append(
            clip.load("RN50x4", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN50x16 is True:
        loguru.logger.info("Load RN50x16")
        clip_models.append(
            clip.load("RN50x16", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN50x64 is True:
        loguru.logger.info("Load RN50x64")
        clip_models.append(
            clip.load("RN50x64", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN101 is True:
        loguru.logger.info("Load RN101")
        clip_models.append(clip.load("RN101", jit=False)[0].eval().requires_grad_(False).to(device))
    return clip_models


def load_lips() -> torch.nn.Module:
    """
    Return LIPS
    """
    loguru.logger.info("Load LIPS")
    device = torch_model_settings.device
    return lpips.LPIPS(net="vgg").to(device)


def load_diffusion_model() -> torch.nn.Module:
    """
    Return Diffusion Model
    """

    device = torch_model_settings.device
    diffusion_model = diffusion_model_settings.diffusion_model.value
    if diffusion_model == DiffusionModelEnum.DIFFUSION_UNCOND_FINTETUNE_008100_512_BY_512.value:
        model = create_model(
            image_size=512,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=True,
            attention_resolutions="32, 16, 8",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=True,
            use_new_attention_order=False,
        )
    elif diffusion_model == DiffusionModelEnum.DIFFUSION_UNCOND_256_BY_256.value:
        model = create_model(
            image_size=256,
            num_channels=256,
            num_res_blocks=2,
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=True,
            attention_resolutions="32, 16, 8",
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=torch_model_settings.use_fp16,
        )
    else:
        loguru.logger.error(f"{diffusion_model} is not supported")
        raise ValueError(f"{diffusion_model} is not supported")
    loguru.logger.info(f"Load {diffusion_model} Model")
    model.load_state_dict(torch.load(f"./pytorch_models/{diffusion_model}.pt", map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if torch_model_settings.use_fp16:
        model.convert_to_fp16()
    return model


def load_diffusion() -> GaussianDiffusion:
    """
    Return diffusion
    """
    loguru.logger.info("Load Diffusion")
    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        timestep_respacing=f"ddim{generation_settings.diffusion_steps.value}",
    )
    return diffusion
