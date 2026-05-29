import torch

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            # Add channel dimension if not present
            clean = batch["clean"].float().to(device)
            noisy = batch["noisy"].float().to(device)
            
            if clean.ndim == 2:
                clean = clean.unsqueeze(1)
            if noisy.ndim == 2:
                noisy = noisy.unsqueeze(1)
                
            output = model(noisy)
            loss = loss_fn(output, clean)
            total_loss += loss.item() * clean.size(0)
            
    return total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
