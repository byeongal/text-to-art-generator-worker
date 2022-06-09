from typing import List, Union

import torch
from pydantic import BaseSettings, Field

from constants import DiffusionModelEnum, DiffusionSamplingModeEnum, DiffusionStepsEnum


class AppSettings(BaseSettings):
    """
    Settings about App
    """

    app_name: str = "Text To Art Generator Worker"
    app_version: str = "0.1.0-dev"
    api_prefix: str = "/api"
    is_debug: bool = True


class IntervalSettings(BaseSettings):
    """
    Settings about interval
    """

    task: int = Field(default=60, description="interval to get task from FireBase")
    health: int = Field(default=10, description="interval to update server health status")


class WorkerSettings(BaseSettings):
    """
    Settings aboout Worker
    """

    worker_id: str = Field(default=None, description="worker id to identify worker")
    worker_key: str = Field(
        default=None, description="worker key to check permissions on worker id"
    )


class FirebaseSettings(BaseSettings):
    """
    Settings about firebase
    """

    func_url: str = Field(description="firebase functions endpoint url")


class TorchModelSettings(BaseSettings):
    """
    Settings about Torch Model
    """

    device: Union[str, torch.device] = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device Information to load Deeplearning Model",
    )
    use_fp16: bool = Field(default=True, description="option to use fp16 for inference")


class ClipModelSettings(BaseSettings):
    """
    Settings about Clip Model
    """

    ViTB32: bool = Field(default=True, description="option to use ViTB32 Model")
    ViTB16: bool = Field(default=True, description="option to use ViTB16 Model")
    ViTL14: bool = Field(default=False, description="option to use ViTL14 Model")
    RN101: bool = Field(default=False, description="option to use RN101 Model")
    RN50: bool = Field(default=True, description="option to use RN50 Model")
    RN50x4: bool = Field(default=False, description="option to use RN50x4 Model")
    RN50x16: bool = Field(default=False, description="option to use RN50x16 Model")
    RN50x64: bool = Field(default=False, description="option to use RN50x64 Model")


class DiffusionModelSettings(BaseSettings):
    """
    Settings about Diffusion Model
    """

    diffusion_model: DiffusionModelEnum = Field(
        default=DiffusionModelEnum.DIFFUSION_UNCOND_FINTETUNE_008100_512_BY_512,
        description="Category of Diffusion Model",
    )
    diffusion_sampling_mode: DiffusionSamplingModeEnum = Field(
        default=DiffusionSamplingModeEnum.DDIM, description="Category of Diffusion Sampling Mode"
    )


class GenerationSettings(BaseSettings):
    """
    Settings about Image Generation
    TODO: Field 로 변경하고 description 추가 하기
    """

    diffusion_steps: DiffusionStepsEnum = Field(
        default=DiffusionStepsEnum.STEP_250, description="Diffusion Steps"
    )
    width: int = Field(default=640, description="Image Width")
    height: int = Field(default=640, description="Image Height")
    clip_denoised: bool = False
    randomize_class: bool = True
    eta: float = 0.8
    cutn_batches: int = 4
    clip_guidance_scale: int = 5000
    tv_scale: int = 0
    range_scale: int = 150
    sat_scale: int = 0
    init_scale: int = 1000
    clamp_grad: bool = True
    clamp_max: float = 0.05
    fuzzy_prompt: bool = False
    rand_mag: float = 0.05

    cut_overview: List[int] = [12] * 400 + [4] * 600
    cut_innercut: List[int] = [4] * 400 + [12] * 600
    cut_ic_pow: int = 1
    cut_icgray_p: List[float] = [0.2] * 400 + [0.0] * 600


app_settings = AppSettings()
interval_settings = IntervalSettings()
worker_settings = WorkerSettings()
firebase_settings = FirebaseSettings()

torch_model_settings = TorchModelSettings()
clip_model_settings = ClipModelSettings()
diffusion_model_settings = DiffusionModelSettings()
generation_settings = GenerationSettings()
