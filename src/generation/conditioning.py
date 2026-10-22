"""Bounded, sequential frozen SDXL prompt encoding for the NF4 backend."""
import gc

import torch

from src.training.cache import checked_tokens


def encode_conditioning(source, prompts: list[str]) -> list[dict]:
    from src.core.device import require_device_lease
    require_device_lease()
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    result = [{} for _ in prompts]
    for index, encoder_class in enumerate((CLIPTextModel, CLIPTextModelWithProjection)):
        folder = "text_encoder" if index == 0 else "text_encoder_2"
        tokenizer_folder = "tokenizer" if index == 0 else "tokenizer_2"
        tokenizer = CLIPTokenizer.from_pretrained(source / tokenizer_folder, local_files_only=True)
        encoder = encoder_class.from_pretrained(source / folder, torch_dtype=torch.float16,
            variant="fp16", local_files_only=True).eval().to("cuda")
        try:
            with torch.inference_mode():
                for i, prompt in enumerate(prompts):
                    ids = checked_tokens(tokenizer, prompt, f"controlled-prompt-{i}")
                    output = encoder(ids.to("cuda"), output_hidden_states=True)
                    result[i][f"hidden_{index}"] = output.hidden_states[-2].cpu().float()
                    if index == 1:
                        result[i]["pooled"] = output.text_embeds.cpu().float()
        finally:
            del encoder
            if "output" in locals(): del output
            gc.collect(); torch.cuda.empty_cache()
    return [{"prompt": torch.cat([r["hidden_0"], r["hidden_1"]], dim=-1), "pooled": r["pooled"]}
            for r in result]
