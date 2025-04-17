import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LoRA utilities
from verl.lora.utils import (
    apply_lora_to_model,
    create_reference_model,
    compute_kl_divergence,
    merge_lora_weights
)


def test_lora_basic():
    """Test basic LoRA functionality."""
    # Load a small model for testing
    model_name = "facebook/opt-125m"
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply LoRA
    lora_config = {
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    }

    logger.info("Applying LoRA to model")
    lora_model = apply_lora_to_model(model, lora_config)

    # Create reference model
    logger.info("Creating reference model")
    ref_model = create_reference_model(lora_model, use_lora=True)

    # Test forward pass
    logger.info("Testing forward pass")
    inputs = tokenizer("Hello, world!", return_tensors="pt")

    with torch.no_grad():
        outputs = lora_model(**inputs)

    logger.info(f"Model output shape: {outputs.logits.shape}")

    # Test KL divergence
    logger.info("Testing KL divergence")
    kl_div = compute_kl_divergence(
        model=lora_model,
        ref_model=ref_model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_lora=True
    )

    logger.info(f"KL divergence: {kl_div}")

    # Test merging weights
    logger.info("Testing weight merging")
    merged_model = merge_lora_weights(lora_model)

    # Test generation with merged model
    logger.info("Testing generation with merged model")
    with torch.no_grad():
        outputs = merged_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")

    logger.info("Basic LoRA test completed successfully!")


def test_lora_integration():
    """Test LoRA integration with VeRL's GRPO."""
    try:
        from verl.lora.algorithms.grpo import LoRAGRPOTrainer
        from verl.lora.config import LoRAConfig

        # Create minimal config
        from types import SimpleNamespace

        config = SimpleNamespace()
        config.model = SimpleNamespace()
        config.model.name_or_path = "facebook/opt-125m"  # Use a small model for testing

        # LoRA config
        lora_config = LoRAConfig(
            use_lora=True,
            lora_rank=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        config.lora = lora_config

        # Training config
        config.train = SimpleNamespace()
        config.train.output_dir = "./lora_test_output"
        config.train.batch_size = 1
        config.train.lr = 5e-5
        config.train.adam_beta1 = 0.9
        config.train.adam_beta2 = 0.999
        config.train.adam_epsilon = 1e-8
        config.train.weight_decay = 0.01
        config.train.max_grad_norm = 1.0

        # Algorithm config
        config.algorithm = SimpleNamespace()
        config.algorithm.kl_coef = 0.1

        # Create trainer
        logger.info("Creating LoRA GRPO trainer")
        trainer = LoRAGRPOTrainer(config)

        # Create dummy batch
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        batch = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["input_ids"],
        }

        # Test compute_loss
        logger.info("Testing loss computation")
        loss = trainer.compute_loss(batch)
        logger.info(f"Loss: {loss.item()}")

        # Test save_checkpoint
        logger.info("Testing checkpoint saving")
        trainer.save_checkpoint(epoch=0, step=0)

        # Test merge_and_save
        logger.info("Testing weight merging and saving")
        output_path = trainer.merge_and_save()
        logger.info(f"Saved merged model to {output_path}")

        logger.info("Integration test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LoRA implementation")
    parser.add_argument("--test_type", type=str, default="basic",
                        choices=["basic", "integration", "all"],
                        help="Type of test to run")
    args = parser.parse_args()

    if args.test_type == "basic" or args.test_type == "all":
        logger.info("=== Running basic LoRA test ===")
        test_lora_basic()

    if args.test_type == "integration" or args.test_type == "all":
        logger.info("\n=== Running LoRA integration test ===")
        test_lora_integration()

    logger.info("All tests completed!")


if __name__ == "__main__":
    main()