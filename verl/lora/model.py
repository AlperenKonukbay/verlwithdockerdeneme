import torch
import logging
from typing import Tuple, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.lora.utils import apply_lora_to_model, create_reference_model

logger = logging.getLogger(__name__)


def create_lora_model(
        model_name_or_path: str,
        lora_config: Dict[str, Any],
        adapter_path: Optional[str] = None,
        dtype: str = "float16",
        device_map: str = "auto",
) -> Tuple[torch.nn.Module, Optional[torch.nn.Module], AutoTokenizer]:
    # Set torch dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    # Get device
    from verl.lora.utils import get_device
    device = get_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Load base model
    logger.info(f"Loading base model: {model_name_or_path}")
    logger.info(f"Using device: {device}")

    # For MPS, load to CPU first then move to MPS
    if device.type == "mps":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        ).to(device)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    # Apply LoRA
    logger.info("Applying LoRA to model")
    model = apply_lora_to_model(
        model=base_model,
        lora_config=lora_config,
        adapter_path=adapter_path
    )

    # Create reference model - ADD THIS PART HERE
    if device.type == "mps":
        # For MPS, create reference model on CPU to avoid compatibility issues
        ref_model = create_reference_model(model, use_lora=True)
        if ref_model is not None:
            ref_model = ref_model.to("cpu")
            logger.info("Reference model moved to CPU for MPS compatibility")
    else:
        ref_model = create_reference_model(model, use_lora=True)

    return model, ref_model, tokenizer