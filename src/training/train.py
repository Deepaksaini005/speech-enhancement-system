import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import random_split

# Add root folder to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cnn_model import CNNModel
from src.data.dataset import VoiceEnhancementDataset
from src.data.dataloader import create_dataloader
from src.training.loss import get_loss_fn
from src.training.validate import validate
from src.data.generate_file_pairs import generate_file_pairs


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        
        clean = batch["clean"].float().to(device)
        noisy = batch["noisy"].float().to(device)
        
        # Add channel dimension if not present
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
            
        output = model(noisy)
        loss = loss_fn(output, clean)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * clean.size(0)
        
    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0


def main(epochs=10, batch_size=8, lr=0.001, duration=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Ensure file pairs CSV exists
    csv_path = ROOT / "data" / "metadata" / "file_pairs.csv"
    if not csv_path.exists() or csv_path.stat().st_size <= 23:
        print("Dataset pairs metadata empty or missing. Auto-generating file_pairs.csv...")
        generate_file_pairs()
        
    # 2. Load dataset
    dataset = VoiceEnhancementDataset(str(csv_path), duration=duration)
    if len(dataset) == 0:
        print("Error: No dataset pairs found. Cannot train model.")
        return
        
    # 3. Train-validation split
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset loaded. Train examples: {train_size}, Val examples: {val_size}")
    
    # 4. Instantiate model, loss, optimizer
    model = CNNModel().to(device)
    loss_fn = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Ensure models directory exists
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float("inf")
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), str(models_dir / "best_model.pth"))
            print(f"--> Saved new best model to models/best_model.pth (Val Loss: {val_loss:.6f})")
            
    # Save final model
    torch.save(model.state_dict(), str(models_dir / "final_model.pth"))
    print("Training complete! Saved final model to models/final_model.pth")


if __name__ == "__main__":
    # Support custom epochs from CLI if desired
    import argparse
    parser = argparse.ArgumentParser(description="Train Voice Denoising Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--duration", type=float, default=1.0, help="Audio segment duration in seconds")
    args = parser.parse_args()
    
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, duration=args.duration)
