import base64
import json
import os
import time

import clip
import loguru
import requests
import torch
import torchvision.transforms.functional as TF
from constants import DiffusionSamplingModeEnum
from fastapi import FastAPI
from settings import (
    diffusion_model_settings,
    firebase_settings,
    generation_settings,
    torch_model_settings,
    worker_settings,
)

from utils.common import clear_memory, set_seed
from utils.model import MakeCutoutsDango, range_loss, spherical_dist_loss, tv_loss


def generate_image(app: FastAPI, text_prompt: str, seed: int, task_id: str):
    """
    test
    """
    device = torch_model_settings.device
    diffusion_sampling_mode = diffusion_model_settings.diffusion_sampling_mode.value

    skip_steps = 10
    batch_size = 1
    side_x = (generation_settings.width // 64) * 64
    side_y = (generation_settings.height // 64) * 64
    clip_denoised = generation_settings.clip_denoised
    randomize_class = generation_settings.randomize_class
    eta = generation_settings.eta
    cutn_batches = generation_settings.cutn_batches
    clip_guidance_scale = generation_settings.clip_guidance_scale
    tv_scale = generation_settings.tv_scale
    range_scale = generation_settings.randomize_class
    sat_scale = generation_settings.sat_scale
    init_scale = generation_settings.init_scale
    clamp_grad = generation_settings.clamp_grad
    clamp_max = generation_settings.clamp_max
    fuzzy_prompt = generation_settings.fuzzy_prompt
    rand_mag = generation_settings.rand_mag

    cut_overview = generation_settings.cut_overview
    cut_innercut = generation_settings.cut_innercut
    cut_ic_pow = generation_settings.cut_ic_pow
    cut_icgray_p = generation_settings.cut_icgray_p

    diffusion_model = app.state.diffusion_model
    diffusion = app.state.diffusion
    clip_models = app.state.clip_models
    normalize = app.state.normalize
    lpips_model = app.state.lpips_model

    set_seed(seed)

    loss_values = []

    frame_prompt = [text_prompt]
    model_stats = []

    for clip_model in clip_models:
        target_embeds, weights = [], []
        model_stat = {
            "clip_model": clip_model,
            "make_cutouts": None,
        }

        for prompt in frame_prompt:
            weight = 1.0
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
            if fuzzy_prompt:
                for _ in range(25):
                    target_embeds.append(
                        (txt + torch.randn(txt.shape).to(device) * rand_mag).clamp(0, 1)
                    )
                    weights.append(weight)
            else:
                target_embeds.append(txt)
                weights.append(weight)

        model_stat["target_embeds"] = torch.cat(target_embeds)
        model_stat["weights"] = torch.tensor(weights, device=device)
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)
    init = None
    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_nan = False
            x = x.detach().requires_grad_()
            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(
                diffusion_model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
            )
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for _ in range(cutn_batches):
                    t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                    input_resolution = 224
                    if (
                        "clip_model" in model_stat
                        and hasattr(model_stat["clip_model"], "visual")
                        and hasattr(model_stat["clip_model"].visual, "input_resolution")
                    ):
                        input_resolution = model_stat["clip_model"].visual.input_resolution

                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=cut_overview[1000 - t_int],
                        InnerCrop=cut_innercut[1000 - t_int],
                        IC_Size_Pow=cut_ic_pow,
                        IC_Grey_P=cut_icgray_p[1000 - t_int],
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0)
                    )
                    dists = dists.view(
                        [
                            cut_overview[1000 - t_int] + cut_innercut[1000 - t_int],
                            n,
                            -1,
                        ]
                    )
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(
                        losses.sum().item()
                    )  # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (
                        torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0]
                        / cutn_batches
                    )
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out["pred_xstart"])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * tv_scale
                + range_losses.sum() * range_scale
                + sat_losses.sum() * sat_scale
            )
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_nan = True
                grad = torch.zeros_like(x)
        if clamp_grad and not x_is_nan:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=clamp_max) / magnitude  # min=-0.02, min=-clamp_max,
        return grad

    if diffusion_sampling_mode == DiffusionSamplingModeEnum.DDIM.value:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive
    cur_t = diffusion.num_timesteps - skip_steps - 1

    if diffusion_sampling_mode == DiffusionSamplingModeEnum.DDIM.value:
        samples = sample_fn(
            diffusion_model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=False,
            skip_timesteps=skip_steps,
            init_image=init,
            randomize_class=randomize_class,
            eta=eta,
        )
    else:
        samples = sample_fn(
            diffusion_model,
            (batch_size, 3, side_y, side_x),
            clip_denoised=clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=False,
            skip_timesteps=skip_steps,
            init_image=init,
            randomize_class=randomize_class,
            order=2,
        )
    image = None
    step = 0
    total_step = generation_settings.diffusion_steps.value - skip_steps
    save_step = max(int(total_step * 0.05), 1)
    if not os.path.exists(f"./output_images/{task_id}"):
        os.makedirs(f"./output_images/{task_id}")
    start = time.time()
    for sample in samples:
        cur_t -= 1
        for image in sample["pred_xstart"]:
            if step % save_step == 0 or step == total_step - 1:
                time_per_step = (time.time() - start) / (step + 1)
                loguru.logger.info(
                    f"[{step}/{total_step} Remaining Time : {time_per_step * (total_step - step - 1)}"
                )
                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                image.save(f"./output_images/{task_id}/{step}.png")
                with open(f"./output_images/{task_id}/{step}.png", "rb") as f:
                    data = f.read()
                base64_str = base64.b64encode(data).decode("utf-8")
                requests.post(
                    f"{firebase_settings.func_url}/updateTaskStatus",
                    headers={"accept": "application/json", "Content-Type": "application/json"},
                    data=json.dumps(
                        {
                            "data": {
                                "workerId": worker_settings.worker_id,
                                "workerKey": worker_settings.worker_key,
                                "taskId": task_id,
                                "image": base64_str,
                                "progress": int((step + 1) / total_step * 100),
                            }
                        }
                    ),
                )

            step += 1
    clear_memory()
    os.rmdir(f"./output_images/{task_id}")
