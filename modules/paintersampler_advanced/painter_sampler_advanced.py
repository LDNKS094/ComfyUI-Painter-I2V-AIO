import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview
import logging

logger = logging.getLogger("Comfyui-PainterSamplerAdvanced")


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=True,
    noise_mask=None,
    callback=None,
    disable_pbar=False,
):
    """Standard ksampler implementation"""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    if disable_noise:
        noise = torch.zeros_like(latent_image)
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask", None)
    if callback is None:
        callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return out


class PainterSamplerAdvanced:
    """
    Advanced Dual-Model Tandem Sampler with separate conditioning for high/low noise phases.

    Designed to work with PainterI2VAdvanced which outputs:
    - high_positive/high_negative: for high-noise phase (motion enhanced)
    - low_positive/low_negative: for low-noise phase (original, color stable)

    This enables proper dual-phase sampling where each phase uses its optimal conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_model": ("MODEL",),
                "low_model": ("MODEL",),
                "add_noise": (["enable", "disable"], {"default": "enable"}),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "high_cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "low_cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                # 高噪声阶段 conditioning
                "high_positive": ("CONDITIONING",),
                "high_negative": ("CONDITIONING",),
                # 低噪声阶段 conditioning
                "low_positive": ("CONDITIONING",),
                "low_negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "switch_at_step": ("INT", {"default": 2, "min": 1, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_leftover_noise": (
                    ["disable", "enable"],
                    {"default": "disable"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "sampling/painter"

    OUTPUT_NODE = False
    ICON = "KSampler"
    WIDTH = 210

    def sample(
        self,
        high_model,
        low_model,
        add_noise,
        noise_seed,
        steps,
        high_cfg,
        low_cfg,
        sampler_name,
        scheduler,
        high_positive,
        high_negative,
        low_positive,
        low_negative,
        latent_image,
        start_at_step,
        switch_at_step,
        end_at_step,
        return_leftover_noise,
    ):
        # 参数标准化
        start_at_step = max(0, start_at_step)
        end_at_step = min(steps, max(start_at_step + 2, end_at_step))
        switch_at_step = max(start_at_step + 1, min(switch_at_step, end_at_step - 1))

        force_full_denoise = return_leftover_noise == "disable"
        disable_noise = add_noise == "disable"

        callback = latent_preview.prepare_callback(high_model, steps)
        disable_pbar = not getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True)

        # 第一阶段：高噪声模型 + 高噪声 conditioning
        if start_at_step < switch_at_step:
            logger.info(
                f"Phase 1: High-noise [{start_at_step}→{switch_at_step}]  cfg={high_cfg}"
            )
            latent_stage1 = latent_image.copy()
            latent_stage1["samples"] = latent_image["samples"].clone()

            samples_stage1 = common_ksampler(
                high_model,
                noise_seed,
                steps,
                high_cfg,
                sampler_name,
                scheduler,
                high_positive,
                high_negative,  # 使用高噪声 conditioning
                latent_stage1,
                denoise=1.0,
                disable_noise=disable_noise,
                start_step=start_at_step,
                last_step=switch_at_step,
                force_full_denoise=False,
                noise_mask=latent_image.get("noise_mask", None),
                callback=callback,
                disable_pbar=disable_pbar,
            )
            current_latent = samples_stage1
        else:
            current_latent = latent_image

        # 第二阶段：低噪声模型 + 低噪声 conditioning
        logger.info(
            f"Phase 2: Low-noise [{switch_at_step}→{end_at_step}]  cfg={low_cfg}"
        )
        samples_final = common_ksampler(
            low_model,
            noise_seed,
            steps,
            low_cfg,
            sampler_name,
            scheduler,
            low_positive,
            low_negative,  # 使用低噪声 conditioning
            current_latent,
            denoise=1.0,
            disable_noise=True,
            start_step=switch_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=current_latent.get("noise_mask", None),
            callback=callback,
            disable_pbar=disable_pbar,
        )

        return (samples_final,)


NODE_CLASS_MAPPINGS = {"PainterSamplerAdvanced": PainterSamplerAdvanced}
NODE_DISPLAY_NAME_MAPPINGS = {"PainterSamplerAdvanced": "Painter Sampler Advanced"}
