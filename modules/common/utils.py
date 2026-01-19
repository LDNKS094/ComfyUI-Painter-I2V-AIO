# modules/common/utils.py
"""
Shared utility functions for PainterAIO nodes.
Extracted from PainterI2V, PainterLongVideo, and Wan22FMLF for reusability.
"""

import torch
import torch.nn.functional as F
import comfy.utils
import comfy.clip_vision
import comfy.latent_formats
import comfy.model_management as mm


def create_video_mask(
    latent_frames: int,
    height: int,
    width: int,
    spacial_scale: int,
    anchor_start: bool = False,
    anchor_end: bool = False,
    device=None,
) -> torch.Tensor:
    """
    Create concat_mask for video conditioning with sub-frame precision.

    Uses [1, 4, T, H, W] format for consistent behavior across all modes.
    This matches the official WanFirstLastFrameToVideo implementation.

    Anchor behavior:
    - Start anchor: locks ALL 4 sub-frames of first latent frame (strong anchoring)
    - End anchor: locks ONLY the last sub-frame of last latent frame (smooth transition)

    Args:
        latent_frames: Number of latent frames
        height: Image height
        width: Image width
        spacial_scale: VAE spatial compression factor (typically 8)
        anchor_start: If True, lock first frame (all 4 sub-frames)
        anchor_end: If True, lock last frame (only last sub-frame)
        device: Target device

    Returns:
        mask tensor of shape [1, 4, latent_frames, H//scale, W//scale]
        Values: 0 = anchor (locked), 1 = generate
    """
    if device is None:
        device = mm.intermediate_device()

    H = height // spacial_scale
    W = width // spacial_scale

    # Create mask at image-frame level (4x temporal resolution)
    mask = torch.ones(
        (1, 1, latent_frames * 4, H, W),
        device=device,
        dtype=torch.float32,
    )

    if anchor_start:
        # Lock first 4 positions = all 4 sub-frames of first latent frame
        mask[:, :, :4] = 0.0

    if anchor_end:
        # Lock only last 1 position = last sub-frame of last latent frame
        mask[:, :, -1:] = 0.0

    # Reshape to [1, 4, latent_frames, H, W]
    # view: [1, 1, T*4, H, W] -> [1, T, 4, H, W]
    # transpose: [1, T, 4, H, W] -> [1, 4, T, H, W]
    mask = mask.view(1, latent_frames, 4, H, W).transpose(1, 2)

    return mask


def apply_motion_amplitude(
    concat_latent: torch.Tensor,
    base_frame_idx: int,
    amplitude: float,
    protect_brightness: bool = True,
) -> torch.Tensor:
    """
    Apply motion amplitude enhancement to concat_latent.

    This fixes slow-motion issues in 4-step LoRAs by amplifying the difference
    between anchor frame and other frames while protecting brightness.

    Args:
        concat_latent: Latent tensor [B, C, T, H, W]
        base_frame_idx: Index of the anchor frame (0 for start, -1 for end)
        amplitude: Motion amplitude multiplier (1.0 = no change, 1.15 = recommended)
        protect_brightness: If True, preserve mean brightness during amplification

    Returns:
        Modified concat_latent tensor
    """
    if amplitude <= 1.0:
        return concat_latent

    # Extract base frame
    if base_frame_idx == 0:
        base_latent = concat_latent[:, :, 0:1]
        other_latent = concat_latent[:, :, 1:]
    else:
        base_latent = concat_latent[:, :, -1:]
        other_latent = concat_latent[:, :, :-1]

    # Compute difference
    diff = other_latent - base_latent

    if protect_brightness:
        # Preserve mean brightness by centering before scaling
        diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
        diff_centered = diff - diff_mean
        scaled_latent = base_latent + diff_centered * amplitude + diff_mean
    else:
        scaled_latent = base_latent + diff * amplitude

    # Clamp to prevent artifacts
    scaled_latent = torch.clamp(scaled_latent, -6, 6)

    # Reconstruct
    if base_frame_idx == 0:
        return torch.cat([base_latent, scaled_latent], dim=2)
    else:
        return torch.cat([scaled_latent, base_latent], dim=2)


def apply_frequency_separation(
    official_latent: torch.Tensor,
    linear_baseline: torch.Tensor,
    boost_scale: float,
    latent_channels: int = 16,
) -> torch.Tensor:
    """
    Apply frequency separation enhancement for FLF2V mode.

    Separates high-frequency (structure/ghosting) from low-frequency (color)
    and boosts only the high-frequency component to reduce ghosting artifacts.

    Args:
        official_latent: Encoded latent from official image sequence
        linear_baseline: Linear interpolation between start and end frames
        boost_scale: High-frequency boost scale (0.0 = no boost, 4.0 = max)
        latent_channels: Number of latent channels (16 for Wan)

    Returns:
        Enhanced latent tensor
    """
    if boost_scale <= 0.001:
        return official_latent

    # Anti-Ghost Vector: diff between official (gray) and linear (PPT-style)
    diff = official_latent - linear_baseline

    h, w = diff.shape[-2], diff.shape[-1]

    # Extract low frequency (color) via downscale + upscale
    low_freq_diff = F.interpolate(
        diff.view(-1, latent_channels, h, w),
        size=(max(1, h // 8), max(1, w // 8)),
        mode="area",
    )
    low_freq_diff = F.interpolate(low_freq_diff, size=(h, w), mode="bilinear")
    low_freq_diff = low_freq_diff.view_as(diff)

    # Extract high frequency (structure/ghosting)
    high_freq_diff = diff - low_freq_diff

    # Boost only high frequency
    return official_latent + (high_freq_diff * boost_scale)


def extract_reference_motion(
    vae,
    video_frames: torch.Tensor,
    width: int,
    height: int,
    target_length: int,
) -> torch.Tensor:
    """
    Extract reference_motion latent from video frames.

    Args:
        vae: VAE model for encoding
        video_frames: Video tensor [T, H, W, C]
        width: Target width
        height: Target height
        target_length: Target number of image frames (will be converted to latent frames)

    Returns:
        reference_motion latent tensor
    """
    device = mm.intermediate_device()

    # Calculate latent frames
    latent_frames = ((target_length - 1) // 4) + 1

    # Take last N frames from reference video
    frames_to_extract = min(target_length, video_frames.shape[0])
    ref_motion = video_frames[-frames_to_extract:].clone()

    # Resize to target dimensions
    ref_motion = comfy.utils.common_upscale(
        ref_motion.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)

    # Pad with gray if insufficient frames
    if ref_motion.shape[0] < target_length:
        gray_fill = (
            torch.ones(
                [target_length, height, width, 3],
                device=device,
                dtype=ref_motion.dtype,
            )
            * 0.5
        )
        gray_fill[-ref_motion.shape[0] :] = ref_motion
        ref_motion = gray_fill

    # Encode and take last latent_frames
    ref_motion_latent = vae.encode(ref_motion[:, :, :, :3])

    # Return the last latent_frames (matching target video length)
    return ref_motion_latent[:, :, -latent_frames:]


def merge_clip_vision_outputs(*outputs):
    """
    Merge multiple CLIP vision outputs by concatenating hidden states.

    Used for FLF2V mode to provide semantic transition guidance.

    Args:
        *outputs: CLIP vision output objects (can include None values)

    Returns:
        Merged CLIP vision output or None if all inputs are None
    """
    valid_outputs = [o for o in outputs if o is not None]

    if not valid_outputs:
        return None

    if len(valid_outputs) == 1:
        return valid_outputs[0]

    # Concatenate penultimate hidden states
    states = torch.cat(
        [o.penultimate_hidden_states for o in valid_outputs],
        dim=-2,
    )

    merged = comfy.clip_vision.Output()
    merged.penultimate_hidden_states = states

    return merged


def apply_clip_vision(clip_vision_output, *conditionings):
    """
    Apply CLIP vision output to multiple conditioning tensors.

    Args:
        clip_vision_output: CLIP vision output object (can be None)
        *conditionings: Conditioning tensors to apply CLIP vision to

    Returns:
        Tuple of modified conditioning tensors (same order as input)
    """
    import node_helpers

    if clip_vision_output is None:
        return conditionings if len(conditionings) > 1 else conditionings[0]

    results = []
    for cond in conditionings:
        results.append(
            node_helpers.conditioning_set_values(
                cond, {"clip_vision_output": clip_vision_output}
            )
        )

    return tuple(results) if len(results) > 1 else results[0]


def get_svi_padding_latent(
    batch_size: int,
    latent_channels: int,
    latent_frames: int,
    height: int,
    width: int,
    spacial_scale: int,
    device=None,
) -> torch.Tensor:
    """
    Get SVI-compatible padding latent (latents_mean).

    For SVI LoRA compatibility, non-anchor frames should be padded with
    Wan21().process_out(zeros) = latents_mean instead of VAE-encoded gray frames.

    Args:
        batch_size: Batch size
        latent_channels: Number of latent channels (16 for Wan)
        latent_frames: Number of latent frames
        height: Image height
        width: Image width
        spacial_scale: VAE spatial compression factor (typically 8)
        device: Target device

    Returns:
        SVI-compatible padding latent tensor
    """
    if device is None:
        device = mm.intermediate_device()

    # Create zeros and apply Wan21 process_out to get latents_mean
    zeros = torch.zeros(
        batch_size,
        latent_channels,
        latent_frames,
        height // spacial_scale,
        width // spacial_scale,
        device=device,
    )

    # process_out converts to latents_mean
    return comfy.latent_formats.Wan21().process_out(zeros)
