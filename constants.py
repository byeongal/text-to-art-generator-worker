from enum import Enum


class ExitStatusEnum(Enum):
    """
    Values for the `__status` in `sys.exit()`.
    """

    REGISTER_WORKER_ERROR = 1


class DiffusionModelEnum(Enum):
    """
    Possible values for the `diffusion_model`.
    """

    DIFFUSION_UNCOND_256_BY_256 = "256x256_diffusion_uncond"
    DIFFUSION_UNCOND_FINTETUNE_008100_512_BY_512 = "512x512_diffusion_uncond_finetune_008100"
    SECONDARY_MODEL_IMAGENET_2 = "secondary_model_imagenet_2"


class DiffusionSamplingModeEnum(Enum):
    """
    Possible values for the `diffusion_sampling_mode`.
    """

    # PLMS = "plms"
    DDIM = "ddim"


class DiffusionStepsEnum(Enum):
    """
    Possible values for the `steps`.
    """

    STEP_25 = 25
    STEP_50 = 50
    STEP_100 = 100
    STEP_150 = 150
    STEP_250 = 250
    STEP_500 = 500
    STEP_1000 = 1000
