#!/usr/bin/env python3
"""
Replicate Training Script for Jewelry VTON (Enhanced Version)

This script initiates a fine-tuning job on Replicate for jewelry virtual try-on
with optimized training parameters for high-quality results.

Key Improvements (v2):
- Increased training steps (2000+ default)
- Better learning rate scheduling with warmup
- Higher resolution training (1024x1024)
- LoRA fine-tuning with optimal rank
- Text encoder training for better concept learning

Environment Variables:
    REPLICATE_API_TOKEN: Required. Your Replicate API token.
    DATASET_URL: Required. URL to the training dataset (zip file with images).
    MODEL_DESTINATION: Optional. Replicate model destination.
    MAX_TRAIN_STEPS: Optional. Number of training steps (default: 2000).
    LEARNING_RATE: Optional. Learning rate (default: 1e-4).
    RESOLUTION: Optional. Training resolution (default: 1024).
    LORA_RANK: Optional. LoRA rank for fine-tuning (default: 32).
"""

import replicate
import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for Replicate SDXL training."""

    # Model settings
    destination: str = "ganeshgowri-asa/naari-jewelry-vton"
    base_model: str = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"

    # Training parameters (improved defaults)
    max_train_steps: int = 2000  # Increased from 1000
    learning_rate: float = 1e-4  # Optimal for SDXL LoRA
    lr_scheduler: str = "cosine_with_restarts"  # Better than constant
    lr_warmup_steps: int = 100  # Warmup for stable training
    lr_num_cycles: int = 3  # Restart cycles for exploration

    # Resolution settings
    resolution: int = 1024  # SDXL native resolution

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32  # Higher rank = more capacity
    lora_alpha: int = 32  # Usually same as rank

    # Text encoder training
    train_text_encoder: bool = True
    text_encoder_lr: float = 5e-5  # Lower than UNet

    # Token and caption settings
    token_string: str = "JEWELRYVTON"
    caption_prefix: str = "a photo of JEWELRYVTON"

    # Training optimizations
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # Use bfloat16 for stability

    # Regularization
    prior_preservation: bool = False
    prior_loss_weight: float = 1.0

    # Saving and logging
    checkpointing_steps: int = 500
    validation_steps: int = 250

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Create config from environment variables with defaults."""
        return cls(
            destination=os.environ.get("MODEL_DESTINATION", cls.destination),
            max_train_steps=int(os.environ.get("MAX_TRAIN_STEPS", cls.max_train_steps)),
            learning_rate=float(os.environ.get("LEARNING_RATE", cls.learning_rate)),
            resolution=int(os.environ.get("RESOLUTION", cls.resolution)),
            lora_rank=int(os.environ.get("LORA_RANK", cls.lora_rank)),
            token_string=os.environ.get("TOKEN_STRING", cls.token_string),
        )


# Training presets for different scenarios
TRAINING_PRESETS = {
    "quick": TrainingConfig(
        max_train_steps=500,
        learning_rate=2e-4,
        lora_rank=16,
        checkpointing_steps=250,
    ),
    "standard": TrainingConfig(
        max_train_steps=2000,
        learning_rate=1e-4,
        lora_rank=32,
        checkpointing_steps=500,
    ),
    "high_quality": TrainingConfig(
        max_train_steps=4000,
        learning_rate=5e-5,
        lora_rank=64,
        lr_warmup_steps=200,
        checkpointing_steps=500,
    ),
    "production": TrainingConfig(
        max_train_steps=6000,
        learning_rate=5e-5,
        lora_rank=64,
        lr_warmup_steps=300,
        train_text_encoder=True,
        text_encoder_lr=2e-5,
        checkpointing_steps=1000,
    ),
}


def validate_environment() -> bool:
    """Validate required environment variables are set."""
    errors = []

    if not os.environ.get("REPLICATE_API_TOKEN"):
        errors.append("REPLICATE_API_TOKEN environment variable is not set")

    if not os.environ.get("DATASET_URL"):
        errors.append("DATASET_URL environment variable is not set")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease set the required environment variables and try again.")
        print("Example:")
        print("  export REPLICATE_API_TOKEN='your-token-here'")
        print("  export DATASET_URL='https://your-dataset-url.zip'")
        return False

    return True


def create_training_input(config: TrainingConfig, dataset_url: str) -> dict:
    """Create the training input dictionary for Replicate API."""
    training_input = {
        # Required inputs
        "input_images": dataset_url,
        "token_string": config.token_string,
        "caption_prefix": config.caption_prefix,

        # Training steps and learning rate
        "max_train_steps": config.max_train_steps,
        "learning_rate": config.learning_rate,

        # Resolution
        "resolution": config.resolution,

        # LoRA configuration
        "use_lora": config.use_lora,
        "lora_rank": config.lora_rank,

        # Text encoder training
        "train_text_encoder": config.train_text_encoder,

        # Optimizations
        "use_8bit_adam": config.use_8bit_adam,
        "gradient_checkpointing": config.gradient_checkpointing,
    }

    # Add learning rate scheduler if supported
    # Note: Replicate's SDXL trainer may have limited scheduler options
    if config.lr_scheduler != "constant":
        training_input["lr_scheduler"] = config.lr_scheduler
        training_input["lr_warmup_steps"] = config.lr_warmup_steps

    return training_input


def create_training_job(config: TrainingConfig) -> object:
    """Create and start a training job on Replicate."""
    dataset_url = os.environ.get("DATASET_URL")

    print("Starting Replicate training job...")
    print()
    print("Configuration:")
    print(f"  Dataset URL: {dataset_url}")
    print(f"  Model destination: {config.destination}")
    print()
    print("Training Parameters:")
    print(f"  Max training steps: {config.max_train_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LR scheduler: {config.lr_scheduler}")
    print(f"  LR warmup steps: {config.lr_warmup_steps}")
    print(f"  Resolution: {config.resolution}x{config.resolution}")
    print()
    print("LoRA Settings:")
    print(f"  Use LoRA: {config.use_lora}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Train text encoder: {config.train_text_encoder}")
    print()
    print("Token Settings:")
    print(f"  Token string: {config.token_string}")
    print(f"  Caption prefix: {config.caption_prefix}")
    print()

    try:
        training_input = create_training_input(config, dataset_url)

        training = replicate.trainings.create(
            version=config.base_model,
            input=training_input,
            destination=config.destination,
        )

        print("Training job created successfully!")
        print(f"  Training ID: {training.id}")
        print(f"  Status: {training.status}")
        print(f"  View at: https://replicate.com/p/{training.id}")
        print()

        return training

    except replicate.exceptions.ReplicateError as e:
        print(f"Replicate API error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error creating training job: {e}")
        raise


def wait_for_training(
    training,
    poll_interval: int = 30,
    max_wait: int = 14400,  # 4 hours for longer training
) -> Optional[bool]:
    """Wait for training to complete with status updates."""
    print(f"Monitoring training progress (polling every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring (training will continue)")
    print()

    start_time = time.time()
    last_logs = ""

    try:
        while True:
            training.reload()
            elapsed = int(time.time() - start_time)
            elapsed_min = elapsed // 60

            # Format status line
            status_line = f"[{elapsed_min}m {elapsed % 60:02d}s] Status: {training.status}"

            # Try to get progress from logs
            if hasattr(training, 'logs') and training.logs:
                # Extract step info from logs
                logs = training.logs
                if logs != last_logs:
                    # Find the last step info
                    import re
                    step_match = re.findall(r"step[:\s]+(\d+)", logs, re.IGNORECASE)
                    if step_match:
                        current_step = step_match[-1]
                        status_line += f" (step {current_step})"
                    last_logs = logs

            print(status_line)

            if training.status == "succeeded":
                print()
                print("=" * 60)
                print("Training completed successfully!")
                print("=" * 60)
                if training.output:
                    print(f"Output model: {training.output}")
                    print()
                    print("To use your trained model:")
                    print(f"  replicate.run('{training.output}', ...)")
                return True

            elif training.status == "failed":
                print()
                print("=" * 60)
                print("Training failed!")
                print("=" * 60)
                if hasattr(training, 'error') and training.error:
                    print(f"Error: {training.error}")
                if hasattr(training, 'logs') and training.logs:
                    print("\nLast logs:")
                    # Print last 20 lines of logs
                    log_lines = training.logs.strip().split('\n')
                    for line in log_lines[-20:]:
                        print(f"  {line}")
                return False

            elif training.status == "canceled":
                print()
                print("Training was canceled")
                return False

            if elapsed > max_wait:
                print()
                print(f"Max wait time ({max_wait}s) exceeded. Training is still running.")
                print(f"Check status at: https://replicate.com/p/{training.id}")
                return None

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print()
        print("Stopped monitoring. Training continues in background.")
        print(f"Check status at: https://replicate.com/p/{training.id}")
        return None


def print_banner():
    """Print script banner."""
    print("=" * 60)
    print("Replicate Training Script for Jewelry VTON (v2)")
    print("=" * 60)
    print()


def list_presets():
    """Print available training presets."""
    print("Available Training Presets:")
    print("-" * 60)
    for name, preset in TRAINING_PRESETS.items():
        print(f"\n{name}:")
        print(f"  Steps: {preset.max_train_steps}")
        print(f"  Learning rate: {preset.learning_rate}")
        print(f"  LoRA rank: {preset.lora_rank}")
        print(f"  Resolution: {preset.resolution}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train Jewelry VTON model on Replicate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard training (2000 steps)
  python train_replicate.py

  # Use a preset
  python train_replicate.py --preset high_quality

  # Custom configuration
  python train_replicate.py --steps 3000 --lr 5e-5 --lora-rank 64

  # Wait for completion
  python train_replicate.py --wait

  # List available presets
  python train_replicate.py --list-presets
        """
    )

    parser.add_argument(
        "--preset", "-p",
        choices=list(TRAINING_PRESETS.keys()),
        help="Use a predefined training preset"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        help="Override max training steps"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        help="Override training resolution"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        help="Override LoRA rank"
    )
    parser.add_argument(
        "--destination", "-d",
        type=str,
        help="Override model destination"
    )
    parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Wait for training to complete"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available training presets"
    )

    args = parser.parse_args()

    print_banner()

    # List presets and exit
    if args.list_presets:
        list_presets()
        return

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Build configuration
    if args.preset:
        config = TRAINING_PRESETS[args.preset]
        print(f"Using preset: {args.preset}")
    else:
        config = TrainingConfig.from_env()
        print("Using standard configuration")

    # Apply overrides
    if args.steps:
        config.max_train_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr
    if args.resolution:
        config.resolution = args.resolution
    if args.lora_rank:
        config.lora_rank = args.lora_rank
    if args.destination:
        config.destination = args.destination

    print()

    # Create training job
    try:
        training = create_training_job(config)
    except Exception as e:
        print(f"\nFailed to create training job: {e}")
        sys.exit(1)

    # Check if we should wait for completion
    wait_for_completion = args.wait or os.environ.get("WAIT_FOR_COMPLETION", "false").lower() == "true"

    if wait_for_completion:
        result = wait_for_training(training)
        if result is False:
            sys.exit(1)
    else:
        print("Training started in background.")
        print()
        print("Options:")
        print("  - Run with --wait flag to monitor progress")
        print("  - Set WAIT_FOR_COMPLETION=true environment variable")
        print(f"  - Check status at: https://replicate.com/p/{training.id}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
