import pytest
import torch

from src.generation.fusion import sample_fused, center_shifts
from src.generation.regions import compile_regions


def tiny_pipe():
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
    model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32,), down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",), cross_attention_dim=8, attention_head_dim=8,
        norm_num_groups=8, addition_embed_type="text_time", addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=20)
    model.eval().requires_grad_(False)
    pipe = StableDiffusionXLPipeline(vae=None, text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=None, unet=model, scheduler=EulerDiscreteScheduler(num_train_timesteps=10),
        add_watermarker=False, force_zeros_for_empty_prompt=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def test_uncontrolled_fusion_matches_actual_diffusers_euler_cfg_and_zero_negative():
    torch.manual_seed(3)
    pipe = tiny_pipe()
    condition = {"prompt": torch.randn(1, 77, 8), "pooled": torch.randn(1, 8)}
    native = pipe(prompt_embeds=condition["prompt"], pooled_prompt_embeds=condition["pooled"],
        num_inference_steps=3, guidance_scale=7, width=64, height=64,
        output_type="latent", generator=torch.Generator().manual_seed(42)).images
    actual = sample_fused(pipe, [condition], 42, 64, 3, 7, torch.float32)
    assert torch.equal(native, actual)


def test_regional_fusion_advances_scheduler_once_per_timestep_and_is_seeded():
    torch.manual_seed(4)
    pipe = tiny_pipe()
    plan = compile_regions({"entities": [{"class_name": "cat"}, {"class_name": "dog"}]}, 64)
    conditions = [{"prompt": torch.randn(1, 77, 8), "pooled": torch.randn(1, 8)} for _ in range(3)]
    backgrounds = torch.zeros(2, 4, 8, 8)
    output = sample_fused(pipe, conditions, 42, 64, 3, 7, torch.float32, plan, backgrounds, 2)
    assert pipe.scheduler.step_index == 3
    repeated = sample_fused(pipe, conditions, 42, 64, 3, 7, torch.float32, plan, backgrounds, 2)
    assert torch.equal(output, repeated)
    assert torch.isfinite(output).all()


def test_fusion_rejects_missing_region_conditioning_and_backgrounds():
    pipe = tiny_pipe()
    condition = {"prompt": torch.zeros(1, 77, 8), "pooled": torch.zeros(1, 8)}
    plan = compile_regions({"entities": [{"class_name": "cat"}, {"class_name": "dog"}]}, 64)
    with pytest.raises(ValueError, match="Missing region"):
        sample_fused(pipe, [condition], 1, 64, 3, 7, torch.float32, plan)
    with pytest.raises(ValueError, match="backgrounds"):
        sample_fused(pipe, [condition]*3, 1, 64, 3, 7, torch.float32, plan, None, 2)


def test_mask_centering_shifts_are_distinct_and_exactly_invertible():
    plan = compile_regions({"entities": [{"class_name": "apple"}] * 3})
    shifts = center_shifts(plan, 64, 64)
    assert shifts[1][1] > 0 and shifts[3][1] < 0
    assert len(set(shifts[1:])) == 3
    pixels = torch.arange(64*64).reshape(1, 1, 64, 64)
    for dy, dx in shifts:
        centered = torch.roll(pixels, (dy, dx), dims=(-2, -1))
        assert torch.equal(torch.roll(centered, (-dy, -dx), dims=(-2, -1)), pixels)
