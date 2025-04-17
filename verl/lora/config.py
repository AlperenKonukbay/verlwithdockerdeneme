from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation)."""
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """Create from dictionary."""
        return cls(
            use_lora=config_dict.get("use_lora", False),
            lora_rank=config_dict.get("lora_rank", 8),
            lora_alpha=config_dict.get("lora_alpha", 32),
            lora_dropout=config_dict.get("lora_dropout", 0.05),
            target_modules=config_dict.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        )


def add_lora_args_to_parser(parser):
    """Add LoRA arguments to ArgumentParser."""
    group = parser.add_argument_group("LoRA")
    group.add_argument("--use_lora", action="store_true", help="Use LoRA for training")
    group.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r)")
    group.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor (alpha)")
    group.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    group.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj,k_proj,o_proj",
        help="Comma-separated list of modules to apply LoRA to, or 'all-linear'"
    )
    group.add_argument("--lora_adapter_path", type=str, help="Path to pre-trained LoRA adapter")
    return parser


def process_lora_args(args) -> LoRAConfig:
    """Process LoRA arguments from ArgumentParser."""
    # Handle target_modules
    if hasattr(args, "target_modules"):
        if args.target_modules == "all-linear":
            target_modules = ["all-linear"]
        else:
            target_modules = args.target_modules.split(",")
    else:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # Create config
    return LoRAConfig(
        use_lora=getattr(args, "use_lora", False),
        lora_rank=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        target_modules=target_modules,
    )