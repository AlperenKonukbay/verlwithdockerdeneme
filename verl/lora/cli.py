import argparse
import logging
import os
import torch
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
import random
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from verl.lora.config import add_lora_args_to_parser, process_lora_args
from verl.lora.algorithms.grpo import LoRAGRPOTrainer

# If DAPO is implemented: from verl.verl.lora.algorithms.dapo import LoRADAPOTrainer

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_lora_parser():
    """Create argument parser with LoRA arguments."""
    parser = argparse.ArgumentParser(description="VeRL with LoRA")

    # Add common arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--algorithm", type=str, default="grpo", choices=["grpo", "dapo"],
                        help="RL algorithm to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"],
                        help="Model precision")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    # Add train arguments
    train_args = parser.add_argument_group("Training")
    train_args.add_argument("--batch_size", type=int, default=4, help="Batch size")
    train_args.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    train_args.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_args.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="Gradient accumulation steps")
    train_args.add_argument("--max_grad_norm", type=float, default=1.0,
                            help="Max gradient norm for clipping")
    train_args.add_argument("--warmup_steps", type=int, default=100,
                            help="Number of warmup steps for lr scheduler")
    train_args.add_argument("--save_steps", type=int, default=500,
                            help="Save checkpoint every X steps")
    train_args.add_argument("--eval_steps", type=int, default=500,
                            help="Evaluate every X steps")
    train_args.add_argument("--log_steps", type=int, default=10,
                            help="Log metrics every X steps")

    # Add data arguments
    data_args = parser.add_argument_group("Data")
    data_args.add_argument("--train_file", type=str, help="Path to training data")
    data_args.add_argument("--val_file", type=str, help="Path to validation data")
    data_args.add_argument("--prompt_column", type=str, default="prompt",
                           help="Column name for prompts")
    data_args.add_argument("--response_column", type=str, default="response",
                           help="Column name for responses")
    data_args.add_argument("--max_seq_length", type=int, default=512,
                           help="Maximum sequence length")

    # Add test data for reward calculation
    data_args.add_argument("--test_prompts", type=str, help="Path to test prompts file")

    # Add LoRA arguments
    add_lora_args_to_parser(parser)

    # Add algorithm-specific arguments
    algo_args = parser.add_argument_group("Algorithm")
    algo_args.add_argument("--kl_coef", type=float, default=0.1,
                           help="KL divergence coefficient")

    # Test mode
    parser.add_argument("--test_mode", action="store_true",
                        help="Run in test mode with minimal data")

    return parser


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""

    def __init__(self, tokenizer, max_length=512, test_mode=False):
        """Initialize dataset."""
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.test_mode = test_mode

        # For test mode, just use a few examples
        if test_mode:
            self.examples = [
                "Hello, my name is",
                "The best way to learn is",
                "Artificial intelligence can",
                "The future of technology is",
                "Climate change is a serious issue because"
            ]
        else:
            # In a real implementation, load data from files
            # For now, just use more examples
            # In SimpleDataset class, expand examples
            self.examples = [
                                "Hello, my name is",
                                "The best way to learn is",
                                "Artificial intelligence can",
                                "The future of technology is",
                                "Climate change is a serious issue because",
                                "The most important skill for a programmer is",
                                "In the next decade, we will see",
                                "The relationship between humans and AI will",
                                "The biggest challenge facing humanity is",
                                "If I could change one thing about the world, it would be",
                                "Education is important because",
                                "Space exploration should focus on",
                                "The internet has transformed society by",
                                "Renewable energy sources include",
                                "Machine learning algorithms work by",
                                "The most influential book I've read is",
                                "Philosophy helps us understand",
                                "Democracy depends on citizens who",
                                "Scientific research should prioritize",
                                "Art and creativity matter because"
                            ] * 5  # Multiply to get 100 examples

    def __len__(self):
        """Return dataset size."""
        return len(self.examples)

    def __getitem__(self, idx):
        """Get dataset item."""
        example = self.examples[idx]

        # Tokenize
        inputs = self.tokenizer(
            example,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create labels (same as input_ids for now - like a simple LM task)
        labels = inputs["input_ids"].clone()

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels.squeeze(0)
        }


def prepare_dataloader(tokenizer, args):
    """Prepare data loader for training/evaluation."""
    # In a real implementation, this would load data from files
    # For demonstration, use a simple dataset
    dataset = SimpleDataset(tokenizer, max_length=args.max_seq_length, test_mode=args.test_mode)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    return train_loader, val_loader


def run_lora_training(args):
    """Run LoRA training with parsed arguments."""
    # Set seed
    set_seed(args.seed)

    # Check device
    device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
    print(f"Using device: {device}")

    # Process LoRA arguments
    lora_config = process_lora_args(args)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create config object
    from types import SimpleNamespace

    config = SimpleNamespace()
    config.lora = lora_config

    # Model config
    config.model = SimpleNamespace()
    config.model.name_or_path = args.model
    config.model.dtype = args.dtype
    config.model.device_map = device  # Use determined device

    # Rest of the function remains the same...

    # Training config
    config.train = SimpleNamespace()
    config.train.output_dir = args.output_dir
    config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    config.train.adam_beta1 = 0.9
    config.train.adam_beta2 = 0.999
    config.train.adam_epsilon = 1e-8
    config.train.weight_decay = 0.01
    config.train.max_grad_norm = args.max_grad_norm
    config.train.epochs = args.epochs
    config.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.train.warmup_steps = args.warmup_steps
    config.train.save_steps = args.save_steps
    config.train.eval_steps = args.eval_steps
    config.train.log_steps = args.log_steps

    # Algorithm config
    config.algorithm = SimpleNamespace()
    config.algorithm.kl_coef = args.kl_coef
    config.algorithm.type = args.algorithm

    # Create trainer based on algorithm
    if args.algorithm.lower() == "grpo":
        trainer = LoRAGRPOTrainer(config)
    elif args.algorithm.lower() == "dapo":
        # If DAPO is implemented
        # trainer = LoRADAPOTrainer(config)
        raise NotImplementedError("DAPO with LoRA not yet implemented")
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Prepare data loaders
    train_loader, val_loader = prepare_dataloader(trainer.tokenizer, args)

    # Training loop
    train(trainer, train_loader, val_loader, config)

    # Finally, merge LoRA weights and save
    output_path = trainer.merge_and_save()
    logger.info(f"Training complete. Merged model saved to: {output_path}")


def train(trainer, train_loader, val_loader, config):
    """Training loop."""
    # Get device
    from verl.lora.utils import get_device
    device = get_device()

    # Initialize variables
    global_step = 0
    best_val_loss = float('inf')

    # Create learning rate scheduler
    total_steps = len(train_loader) * config.train.epochs // config.train.gradient_accumulation_steps

    if hasattr(trainer, 'scheduler') and trainer.scheduler is None:
        trainer.scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=config.train.warmup_steps,
            num_training_steps=total_steps
        )

    # Training loop
    logger.info("Starting training...")
    for epoch in range(config.train.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.train.epochs}")

        # Training
        trainer.model.train()
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        # Find this part in the train function
        for step, batch in enumerate(progress_bar):
            # No need to move batch to device here, it's handled in train_step

            # Forward pass
            metrics = trainer.train_step(batch)
            loss = metrics["loss"]

            # Update progress bar
            progress_bar.set_postfix(loss=loss)

            # Update metrics
            train_loss += loss
            train_steps += 1
            global_step += 1

            # Log, save, and evaluate
            if global_step % config.train.log_steps == 0:
                logger.info(f"Step {global_step}: loss = {loss:.4f}")

            if global_step % config.train.save_steps == 0:
                logger.info(f"Saving checkpoint at step {global_step}")
                trainer.save_checkpoint(epoch=epoch, step=global_step)

            if global_step % config.train.eval_steps == 0:
                # Evaluate
                val_loss = evaluate(trainer, val_loader)
                logger.info(f"Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                    trainer.save_checkpoint(epoch=epoch, step=global_step)

                # Back to training mode
                trainer.model.train()

        # Epoch complete
        avg_train_loss = train_loss / train_steps
        logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_train_loss:.4f}")

        # Evaluate at end of epoch
        val_loss = evaluate(trainer, val_loader)
        logger.info(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")

        # Save checkpoint
        trainer.save_checkpoint(epoch=epoch, step=global_step)

    logger.info("Training complete.")


def evaluate(trainer, val_loader):
    """Evaluate model."""
    logger.info("Evaluating model...")
    trainer.model.eval()

    total_loss = 0.0
    steps = 0

    # Get device
    from verl.lora.utils import get_device
    device = get_device()

    # For MPS, fall back to CPU for evaluation
    if device.type == "mps":
        logger.info("Using CPU for evaluation to avoid MPS compatibility issues")
        eval_device = "cpu"
    else:
        eval_device = device

    try:
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to evaluation device
                batch = {k: v.to(eval_device) for k, v in batch.items()}

                # Temporarily move model to evaluation device if needed
                if eval_device != device:
                    original_device = next(trainer.model.parameters()).device
                    trainer.model = trainer.model.to(eval_device)

                # Compute loss
                outputs = trainer.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss

                # Move model back if needed
                if eval_device != device:
                    trainer.model = trainer.model.to(original_device)

                # Update metrics
                total_loss += loss.item()
                steps += 1
    except Exception as e:
        logger.warning(f"Evaluation failed with error: {e}")
        logger.info("Skipping evaluation and continuing with training")
        return float('inf')  # Return high loss so it doesn't get saved as best model

    # Return average loss or placeholder if no steps completed
    return total_loss / max(steps, 1)


def main():
    """Main entry point for LoRA CLI."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Parse arguments
    parser = create_lora_parser()
    args = parser.parse_args()

    # Run training
    run_lora_training(args)


if __name__ == "__main__":
    main()