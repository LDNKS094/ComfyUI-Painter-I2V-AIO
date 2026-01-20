import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview
import logging
from comfy_api.latest import io

logger = logging.getLogger("Comfyui-PainterSampler")


# 官方 common_ksampler 副本（保持原样）
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


class PainterSampler(io.ComfyNode):
    """
    Dual-Model Tandem Sampler: 100% replicates the generation effect of the official KSamplerAdvanced,
    with only dual-model input integrated.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PainterSampler",
            display_name="Painter Sampler",
            category="sampling/painter",
            inputs=[
                io.Model.Input("high_model"),
                io.Model.Input("low_model"),
                io.Combo.Input(
                    "add_noise", options=["enable", "disable"], default="enable"
                ),
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("high_cfg", default=8.0, min=0.0, max=100.0, step=0.01),
                io.Float.Input("low_cfg", default=8.0, min=0.0, max=100.0, step=0.01),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.KSampler.SAMPLERS
                ),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("start_at_step", default=0, min=0, max=10000),
                io.Int.Input("switch_at_step", default=2, min=1, max=10000),
                io.Int.Input("end_at_step", default=10000, min=0, max=10000),
                io.Combo.Input(
                    "return_leftover_noise",
                    options=["disable", "enable"],
                    default="disable",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(
        cls,
        high_model,
        low_model,
        add_noise,
        noise_seed,
        steps,
        high_cfg,
        low_cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        switch_at_step,
        end_at_step,
        return_leftover_noise,
    ) -> io.NodeOutput:
        start_at_step = max(0, start_at_step)
        end_at_step = min(steps, max(start_at_step + 2, end_at_step))
        switch_at_step = max(start_at_step + 1, min(switch_at_step, end_at_step - 1))

        force_full_denoise = return_leftover_noise == "disable"
        disable_noise = add_noise == "disable"

        callback = latent_preview.prepare_callback(high_model, steps)
        disable_pbar = not getattr(comfy.utils, "PROGRESS_BAR_ENABLED", True)

        # 第一阶段：高噪声模型
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
                positive,
                negative,
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

        # 第二阶段：低噪声模型
        logger.info(
            f"Phase 2: Low-noise [{switch_at_step}→{end_at_step}]  cfg={low_cfg}"
        )
        callback_low = latent_preview.prepare_callback(low_model, steps)
        samples_final = common_ksampler(
            low_model,
            noise_seed,
            steps,
            low_cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            current_latent,
            denoise=1.0,
            disable_noise=True,
            start_step=switch_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=current_latent.get("noise_mask", None),
            callback=callback_low,
            disable_pbar=disable_pbar,
        )

        return io.NodeOutput(samples_final)
