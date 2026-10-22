"""Load frozen SDXL weights and train only PEFT attention adapters."""
import torch
from pathlib import Path

from src.training.common import TrainingConfig
from src.core.device import require_device_lease


def training_model_source(model: str) -> str:
    """Training never downloads a new checkpoint implicitly."""
    local = Path(model).expanduser()
    if local.is_dir():
        return str(local.resolve())
    from huggingface_hub import hf_hub_download
    marker = hf_hub_download(model, "model_index.json", local_files_only=True)
    return str(Path(marker).parent)


def load_trainable_unet(config: TrainingConfig):
    require_device_lease()
    from diffusers import BitsAndBytesConfig, UNet2DConditionModel
    from peft import LoraConfig

    if not torch.cuda.is_available():
        raise RuntimeError("This SDXL training profile requires a CUDA GPU")
    free, total = torch.cuda.mem_get_info()
    if total / 1e9 < config.max_vram_gb or free / 1e9 < config.min_free_vram_gb:
        raise RuntimeError("Insufficient free GPU memory for the configured training profile")
    if config.method == "lora" and total / 1e9 <= 6.5:
        raise RuntimeError("The 6 GB training profile requires QLoRA; plain SDXL LoRA is not validated in this budget")
    torch.use_deterministic_algorithms(config.deterministic, warn_only=False)
    if config.deterministic:
        # PyTorch's memory-efficient SDP backward warns about nondeterminism.
        # Native flash or math attention can run with strict algorithm checks.
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"torch_dtype": dtype, "variant": "fp16", "use_safetensors": True,
              "low_cpu_mem_usage": True}
    if config.method == "qlora":
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
            device_map={"": "cuda:0"},
        )
    unet = UNet2DConditionModel.from_pretrained(
        training_model_source(config.model), subfolder="unet", local_files_only=True, **kwargs)
    if config.method == "lora":
        unet.to("cuda")
    if config.method == "qlora":
        import bitsandbytes as bnb
        if not any(isinstance(module, bnb.nn.Linear4bit) for module in unet.modules()):
            raise RuntimeError("QLoRA requested but no NF4 linear modules were loaded")
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=config.rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
        target_modules=config.target_modules, init_lora_weights="gaussian"))
    unet.enable_gradient_checkpointing()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were created")
    for parameter in trainable:
        parameter.data = parameter.data.float()
    frozen = [name for name, p in unet.named_parameters()
              if p.requires_grad and "lora_" not in name]
    if frozen:
        raise RuntimeError("Unexpected base parameters marked trainable")
    unet.train()
    return unet, dtype
