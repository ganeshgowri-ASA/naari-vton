"""
Replicate Training Script for Jewelry VTON

This script initiates a fine-tuning job on Replicate for jewelry virtual try-on.

Environment Variables:
    REPLICATE_API_TOKEN: Required. Your Replicate API token.
    DATASET_URL: Required. URL to the training dataset (zip file with images).
    MODEL_DESTINATION: Optional. Replicate model destination (default: ganeshgowri-asa/naari-jewelry-vton).
    MAX_TRAIN_STEPS: Optional. Number of training steps (default: 1000).
"""
import replicate
import os
import sys
import time


# Training configuration
DEFAULT_MODEL_DESTINATION = "ganeshgowri-asa/naari-jewelry-vton"
DEFAULT_MAX_TRAIN_STEPS = 1000
DEFAULT_TOKEN_STRING = "JEWELRYVTON"
DEFAULT_CAPTION_PREFIX = "a photo of JEWELRYVTON jewelry"

# Base model for fine-tuning (SDXL)
SDXL_VERSION = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"


def validate_environment():
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


def create_training_job():
    """Create and start a training job on Replicate."""
    dataset_url = os.environ.get("DATASET_URL")
    destination = os.environ.get("MODEL_DESTINATION", DEFAULT_MODEL_DESTINATION)
    max_steps = int(os.environ.get("MAX_TRAIN_STEPS", DEFAULT_MAX_TRAIN_STEPS))

    print(f"Starting Replicate training job...")
    print(f"  Dataset URL: {dataset_url}")
    print(f"  Model destination: {destination}")
    print(f"  Max training steps: {max_steps}")
    print(f"  Token string: {DEFAULT_TOKEN_STRING}")
    print()

    try:
        training = replicate.trainings.create(
            version=SDXL_VERSION,
            input={
                "input_images": dataset_url,
                "token_string": DEFAULT_TOKEN_STRING,
                "caption_prefix": DEFAULT_CAPTION_PREFIX,
                "max_train_steps": max_steps,
            },
            destination=destination,
        )

        print(f"Training job created successfully!")
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


def wait_for_training(training, poll_interval=30, max_wait=7200):
    """Optionally wait for training to complete with status updates."""
    print(f"Monitoring training progress (polling every {poll_interval}s)...")
    print("Press Ctrl+C to stop monitoring (training will continue)")
    print()

    start_time = time.time()

    try:
        while True:
            training.reload()
            elapsed = int(time.time() - start_time)

            print(f"[{elapsed}s] Status: {training.status}")

            if training.status == "succeeded":
                print(f"\nTraining completed successfully!")
                print(f"  Output model: {training.output}")
                return True
            elif training.status == "failed":
                print(f"\nTraining failed!")
                if hasattr(training, 'error') and training.error:
                    print(f"  Error: {training.error}")
                return False
            elif training.status == "canceled":
                print(f"\nTraining was canceled")
                return False

            if elapsed > max_wait:
                print(f"\nMax wait time ({max_wait}s) exceeded. Training is still running.")
                print(f"Check status at: https://replicate.com/p/{training.id}")
                return None

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(f"\nStopped monitoring. Training continues in background.")
        print(f"Check status at: https://replicate.com/p/{training.id}")
        return None


def main():
    """Main entry point for training script."""
    print("=" * 60)
    print("Replicate Training Script for Jewelry VTON")
    print("=" * 60)
    print()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Create training job
    try:
        training = create_training_job()
    except Exception as e:
        print(f"\nFailed to create training job: {e}")
        sys.exit(1)

    # Check if we should wait for completion
    wait_for_completion = os.environ.get("WAIT_FOR_COMPLETION", "false").lower() == "true"

    if wait_for_completion:
        result = wait_for_training(training)
        if result is False:
            sys.exit(1)
    else:
        print("Training started. To monitor progress, run with WAIT_FOR_COMPLETION=true")
        print(f"Or check status at: https://replicate.com/p/{training.id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
