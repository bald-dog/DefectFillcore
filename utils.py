import torch
import os
from typing import Optional, Dict, Any

def save_checkpoint(model, optimizer, step, path):
    """
    Save model checkpoint
    
    Args:
        model: The DefectFill model
        optimizer: Optimizer state
        step: Current training step
        path: Path to save checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model states
    checkpoint = {
        "step": step,
        "text_encoder_lora": {k: v for k, v in model.pipeline.text_encoder.state_dict().items() if "lora" in k},
        "unet_lora": {k: v for k, v in model.pipeline.unet.state_dict().items() if "lora" in k},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path) -> int:
    """
    Load model checkpoint
    
    Args:
        model: The DefectFill model
        optimizer: Optimizer to load state into
        path: Path to checkpoint
        
    Returns:
        Current step from checkpoint
    """
    # Check if checkpoint exists
    if not os.path.exists(path):
        print(f"Checkpoint {path} not found, starting from scratch")
        return 0
    
    # Load checkpoint
    checkpoint = torch.load(path)
    
    # Load text encoder LoRA weights
    text_encoder_sd = model.pipeline.text_encoder.state_dict()
    for k, v in checkpoint["text_encoder_lora"].items():
        if k in text_encoder_sd:
            text_encoder_sd[k] = v
    model.pipeline.text_encoder.load_state_dict(text_encoder_sd)
    
    # Load UNet LoRA weights
    unet_sd = model.pipeline.unet.state_dict()
    for k, v in checkpoint["unet_lora"].items():
        if k in unet_sd:
            unet_sd[k] = v
    model.pipeline.unet.load_state_dict(unet_sd)
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint["step"]