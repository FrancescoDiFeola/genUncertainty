import os
import json
import torch
from datetime import datetime
import re

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, model_name="autoencoder"):
    """
    Save model checkpoint in district-specific directory.
    
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Base directory to save the checkpoint.
        opt (TrainOptions): Options object containing the district name.
        model_name (str): Name of the model (default: "autoencoder").
    """
    # Create a directory for the district based on the opt.district
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch + 1}.pth")
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    torch.save(checkpoint_data, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    # Update checkpoint_info.json for the specific district
    update_checkpoint_info(epoch, checkpoint_path, checkpoint_dir, model_name)


def update_checkpoint_info(epoch, checkpoint_path, checkpoint_dir, model_name="autoencoder"):
    """
    Update and save checkpoint metadata in `checkpoint_info.json` for each district.
    
    Args:
        epoch (int): Current epoch number.
        checkpoint_path (str): Path to the saved checkpoint.
        district_checkpoint_dir (str): Directory where checkpoints are saved for the district.
        model_name (str): Name of the model (default: "autoencoder").
    """
    checkpoint_info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")

    # Load existing info if available
    if os.path.exists(checkpoint_info_path):
        with open(checkpoint_info_path, "r") as f:
            checkpoint_info = json.load(f)
    else:
        checkpoint_info = {}

    # Update checkpoint info
    checkpoint_info["last_saved_epoch"] = epoch
    checkpoint_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_info.setdefault("saved_checkpoints", []).append(checkpoint_path)

    # Save JSON file
    with open(checkpoint_info_path, "w") as f:
        json.dump(checkpoint_info, f, indent=4)

    print(f"[INFO] Checkpoint info updated: {checkpoint_info_path}")


def load_checkpoint(model, optimizer, checkpoint_dir, model_name="autoencoder", device="cuda", strict=True):
    """
    Load model and optimizer state from the latest checkpoint for a specific district.
    
    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to load state.
        checkpoint_dir (str): Base directory containing checkpoints.
        opt (TrainOptions): Options object containing the district name.
        model_name (str): Name of the model (default: "autoencoder").
        device (str): Device to load the model onto (default: "cuda").
    
    Returns:
        int: Last saved epoch number.
    """
    checkpoint_info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")

    if not os.path.exists(checkpoint_info_path):
        print("[WARNING] No checkpoint info found! Starting training from scratch.")
        return 0  # Start from epoch 0

    # Load checkpoint info
    with open(checkpoint_info_path, "r") as f:
        checkpoint_info = json.load(f)

    if "saved_checkpoints" not in checkpoint_info or len(checkpoint_info["saved_checkpoints"]) == 0:
        print("[WARNING] No saved checkpoints found in JSON!")
        return 0

    # INIZIO CORREZIONE CODICE
    model_checkpoints = [
        cp for cp in checkpoint_info["saved_checkpoints"]
        if model_name in os.path.basename(cp)
    ]

    if not model_checkpoints:
        print(f"[WARNING] No saved checkpoints found for model '{model_name}' in JSON!")
        return 0

    latest_checkpoint = model_checkpoints[-1]

    print(f"[INFO] Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as e:
            print(f"[WARNING] Could not load optimizer state: {e}. Optimizer state will be re-initialized.")
    elif optimizer is not None:
        print("[WARNING] Optimizer state not found in checkpoint. Optimizer state will be re-initialized.")

    epoch_match = re.search(r'epoch(\d+)', os.path.basename(latest_checkpoint))
    if epoch_match:
        return int(epoch_match.group(1)) + 1  # Riprendi dall'epoca successiva
    else:
        print(f"[WARNING] Could not parse epoch from checkpoint name: {latest_checkpoint}. Returning epoch 0.")
        return 0

    """
    latest_checkpoint = checkpoint_info["saved_checkpoints"][-1]  # Load latest checkpoint
    print(f"[INFO] Loading checkpoint: {latest_checkpoint}")

    # Load checkpoint data
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint_info["last_saved_epoch"] + 1  # Resume from next epoch
    """

def save_checkpoint_BB(model, optimizer, epoch, checkpoint_dir, model_config, model_name="autoencoder"):
    """
    Save model checkpoint in district-specific directory.
    
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Current epoch number.
        checkpoint_dir (str): Base directory to save the checkpoint.
        opt (TrainOptions): Options object containing the district name.
        model_name (str): Name of the model (default: "autoencoder").
    """
    # Create a directory for the district based on the opt.district
    district_checkpoint_dir = checkpoint_dir
    os.makedirs(district_checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(district_checkpoint_dir, f"{model_name}_epoch{epoch}.pth")
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")
    # Optional: update metadata tracking
    update_checkpoint_info(epoch, checkpoint_path, district_checkpoint_dir, model_name)

# Function to load the autoencoder checkpoint
def load_autoencoder_checkpoint(autoencoder, checkpoint_path, device="device"):
    """Loads pre-trained weights into the autoencoder. If no checkpoint exists, initializes the model from scratch."""
    
    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_path):
        # If the checkpoint exists, load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path} for the autoencoder.")
        return checkpoint  # Return the checkpoint if needed (e.g., for getting epoch info)
    else:
        # If the checkpoint does not exist, initialize the model from scratch
        print(f"No checkpoint found at {checkpoint_path}. Initializing the autoencoder from scratch.")
    
        return None  # Return None or some indication that training starts from scratch