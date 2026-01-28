"""
Replicate Training Script for Jewelry VTON
"""
import replicate
import os

def create_training_job():
    training = replicate.trainings.create(
        version="stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={
            "input_images": "https://your-dataset-url.zip",
            "token_string": "JEWELRYVTON",
            "caption_prefix": "a photo of JEWELRYVTON jewelry",
            "max_train_steps": 1000,
        },
        destination="ganeshgowri/naari-jewelry-vton"
    )
    print(f"Training started: {training.id}")
    return training

if __name__ == "__main__":
    print("Replicate Training Script Ready")
