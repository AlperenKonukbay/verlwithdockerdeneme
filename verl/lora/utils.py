import torch
import logging
import os
from typing import List, Optional, Union, Dict, Any

try:
    from peft import LoraConfig, get_peft_model, PeftModel

    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: PEFT library not found. Install with 'pip install peft'")

def get_device():
    """Get the best available device (MPS for Mac GPU, CPU otherwise)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

logger = logging.getLogger(__name__)


def resolve_target_modules(model, target_modules: List[str]) -> List[str]:
    """
    Convert shorthand module names to actual module names in the model.

    Args:
        model: The base model
        target_modules: List of target module names or ["all-linear"]

    Returns:
        List of resolved module names
    """
    if len(target_modules) == 1 and target_modules[0] == "all-linear":
        # Find all linear layers
        result = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Extract the final part of the name
                parts = name.split(".")
                result.append(parts[-1])
        # Remove duplicates
        return list(set(result))
    return target_modules


def apply_lora_to_model(
        model: torch.nn.Module,
        lora_config: Dict[str, Any],
        adapter_path: Optional[str] = None,
) -> torch.nn.Module:
    """
    Apply LoRA to a model.

    Args:
        model: The base model
        lora_config: LoRA configuration parameters
        adapter_path: Path to load a pre-trained adapter

    Returns:
        Model with LoRA adapters
    """
    if not HAS_PEFT:
        logger.warning("PEFT library not available, skipping LoRA application")
        return model

    # Extract LoRA parameters
    lora_rank = lora_config.get("lora_rank", 8)
    lora_alpha = lora_config.get("lora_alpha", 32)
    lora_dropout = lora_config.get("lora_dropout", 0.05)
    target_modules = lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Resolve target modules
    resolved_modules = resolve_target_modules(model, target_modules)

    # Create LoRA config
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=resolved_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    lora_model = get_peft_model(model, peft_config)

    # Load adapter if provided
    if adapter_path:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        lora_model.load_adapter(adapter_path)

    # Add this code here to count and log parameters
    def count_parameters(model):
        """Count trainable and total parameters in a model."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

    trainable_params, total_params = count_parameters(lora_model)
    reduction_factor = 100 * (1 - trainable_params / total_params)
    # Replace the logger statements with these:
    print("\n" + "=" * 50)
    print("LORA PARAMETER STATISTICS")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    print(f"Parameter reduction:  {reduction_factor:.2f}%")
    print("=" * 50 + "\n")

    return lora_model


def create_reference_model(model, use_lora: bool = False) -> Optional[torch.nn.Module]:
    """
    Create reference model for KL divergence calculation.

    Args:
        model: Policy model (potentially with LoRA)
        use_lora: Whether policy model uses LoRA

    Returns:
        Reference model or None if using disable_adapter approach
    """
    # If we have a LoRA model that supports adapter disabling, we can reuse the same model
    if use_lora and hasattr(model, "disable_adapter") and hasattr(model, "enable_adapter"):
        logger.info("Using same model with adapter disabling for KL reference")
        return None

    logger.info("Creating separate reference model for KL divergence")

    # Get base model path
    if hasattr(model, "get_base_model") and callable(model.get_base_model):
        # For PEFT models, get base model
        base_model = model.get_base_model()
        base_model_path = base_model.config._name_or_path
    else:
        # Regular model
        base_model_path = model.config._name_or_path if hasattr(model, "config") else None

    if not base_model_path:
        logger.error("Could not determine base model path for reference model")
        return model  # Return original model as fallback

    # Import here to avoid circular imports
    from transformers import AutoModelForCausalLM

    # Create reference model with same config
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=next(model.parameters()).dtype
    )

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def compute_kl_divergence(
        model: torch.nn.Module,
        ref_model: Optional[torch.nn.Module],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_lora: bool = False,
        kl_coef: float = 0.1,
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference model.

    Args:
        model: Policy model (potentially with LoRA)
        ref_model: Reference model or None if using disable_adapter
        input_ids: Input token IDs
        attention_mask: Attention mask
        use_lora: Whether policy model uses LoRA
        kl_coef: KL divergence coefficient

    Returns:
        KL divergence loss
    """
    # Check if using MPS and fall back to CPU if needed
    is_mps = input_ids.device.type == "mps"

    # When using MPS, we'll use CPU for reference model for better compatibility
    if is_mps and ref_model is not None:
        # Move ref_model to CPU
        ref_model = ref_model.to("cpu")

        # We'll also need to move input tensors to CPU for ref_model
        cpu_input_ids = input_ids.to("cpu")
        cpu_attention_mask = attention_mask.to("cpu") if attention_mask is not None else None

        # Get reference logits on CPU
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=cpu_input_ids,
                attention_mask=cpu_attention_mask,
            )
            ref_logits = ref_outputs.logits.to(input_ids.device)  # Move back to MPS
    elif use_lora and ref_model is None and hasattr(model, "disable_adapter"):
        # MPS device with adapter disabling
        model.disable_adapter()
        with torch.no_grad():
            ref_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits
        model.enable_adapter()
    else:
        # Regular path (CPU or other devices)
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits

    # Get policy logits
    policy_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    policy_logits = policy_outputs.logits

    # Compute KL divergence: KL(policy || ref)
    policy_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
    ref_probs = torch.nn.functional.softmax(ref_logits, dim=-1)
    kl_div = torch.nn.functional.kl_div(
        policy_log_probs,
        ref_probs,
        reduction="batchmean",
        log_target=False,
    )

    return kl_div * kl_coef


def save_lora_checkpoint(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[object],
        epoch: int,
        step: int,
        checkpoint_dir: str,
        save_optimizer_state: bool = True,
) -> None:
    """
    Save LoRA checkpoint.

    Args:
        model: LoRA model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current step
        checkpoint_dir: Directory to save checkpoint
        save_optimizer_state: Whether to save optimizer state
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save adapter weights
    if hasattr(model, "save_pretrained") and callable(model.save_pretrained):
        logger.info(f"Saving LoRA adapter to {checkpoint_dir}")
        model.save_pretrained(checkpoint_dir)
    else:
        logger.warning("Model doesn't have save_pretrained method, can't save adapter weights")

    # Save training state
    if save_optimizer_state and optimizer is not None:
        train_state = {
            'epoch': epoch,
            'step': step,
            'optimizer': optimizer.state_dict(),
        }

        if scheduler is not None:
            train_state['scheduler'] = scheduler.state_dict()

        logger.info(f"Saving training state to {checkpoint_dir}/training_state.pt")
        torch.save(train_state, os.path.join(checkpoint_dir, "training_state.pt"))


def load_lora_checkpoint(
        base_model: torch.nn.Module,
        lora_config: Dict[str, Any],
        adapter_path: str,
        training_state_path: Optional[str] = None,
) -> tuple:
    """
    Load LoRA checkpoint.

    Args:
        base_model: Base model
        lora_config: LoRA configuration
        adapter_path: Path to LoRA adapter
        training_state_path: Path to training state

    Returns:
        (lora_model, training_state)
    """
    # Apply LoRA to base model
    lora_model = apply_lora_to_model(
        model=base_model,
        lora_config=lora_config,
    )

    # Load adapter weights
    if hasattr(lora_model, "load_adapter") and callable(lora_model.load_adapter):
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        lora_model.load_adapter(adapter_path)
    else:
        logger.warning("Model doesn't have load_adapter method, can't load adapter weights")

    # Load training state if path provided
    training_state = None
    if training_state_path and os.path.exists(training_state_path):
        logger.info(f"Loading training state from {training_state_path}")
        training_state = torch.load(training_state_path, map_location="cpu")

    return lora_model, training_state


def merge_lora_weights(
        model: torch.nn.Module,
        save_path: Optional[str] = None,
) -> torch.nn.Module:
    """
    Merge LoRA weights into base model.

    Args:
        model: LoRA model
        save_path: Path to save merged model

    Returns:
        Merged model
    """
    if not hasattr(model, "merge_and_unload") or not callable(model.merge_and_unload):
        logger.warning("Model doesn't have merge_and_unload method, can't merge LoRA weights")
        return model

    # Merge weights
    logger.info("Merging LoRA weights into base model")
    merged_model = model.merge_and_unload()

    # Save merged model if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving merged model to {save_path}")
        merged_model.save_pretrained(save_path)

    return merged_model