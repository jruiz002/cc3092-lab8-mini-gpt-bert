import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math

from model import MiniGPT
from data import TinyShakespeareDataset

def main():
    # Hyperparameters from Task 1.1
    block_size = 128
    d_model = 128
    n_heads = 4
    n_layers = 3
    batch_size = 64
    lr = 3e-4
    epochs = 10
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")

    # Data
    print("Loading data...")
    # Dataset will be downloaded to root directory
    dataset = TinyShakespeareDataset(data_dir='..', block_size=block_size)
    
    print("-" * 50)
    print("EVIDENCIA TASK 1.1")
    print(f"Tamaño del vocabulario: {dataset.vocab_size} caracteres")
    print(f"Tokens en split de entrenamiento (90%): {len(dataset.train_data)}")
    print(f"Tokens en split de validación (10%): {len(dataset.val_data)}")
    
    X_sample, Y_sample = dataset.get_batch('train', batch_size=1)
    print("\nEjemplo de desplazamiento de batch (1 secuencia, primeros 10 tokens):")
    print(f"Input (X):  {X_sample[0, :10].tolist()} -> {repr(dataset.decode(X_sample[0, :10].tolist()))}")
    print(f"Target (Y): {Y_sample[0, :10].tolist()} -> {repr(dataset.decode(Y_sample[0, :10].tolist()))}")
    print("-" * 50)
    
    # Model
    print("Initializing model...")
    model = MiniGPT(V=dataset.vocab_size, 
                    d_model=d_model, 
                    n_heads=n_heads, 
                    n_layers=n_layers, 
                    block_size=block_size)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists for plotting
    train_losses = []
    val_losses = []

    # Number of iterations per epoch
    # 1 epoch ~ entire dataset processed once
    iters_per_epoch = len(dataset.train_data) // (batch_size * block_size)
    print(f"Iterations per epoch: {iters_per_epoch}")

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for _ in range(iters_per_epoch):
            X, Y = dataset.get_batch('train', batch_size)
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(X, Y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / iters_per_epoch
        
        # Evaluate validation loss
        val_losses_batch = []
        model.eval()
        with torch.no_grad():
            # Evaluate over ~50 batches
            eval_iters = 50
            for _ in range(eval_iters):
                X_val, Y_val = dataset.get_batch('val', batch_size)
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                _, val_loss = model(X_val, Y_val)
                val_losses_batch.append(val_loss.item())
        avg_val_loss = sum(val_losses_batch) / len(val_losses_batch)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Final perplexity
    final_val_loss = val_losses[-1]
    val_perplexity = math.exp(final_val_loss)
    print(f"\nFinal Validation Perplexity: {val_perplexity:.4f}")

    # Save results
    os.makedirs('../resultados', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), '../resultados/minigpt_weights.pth')
    print("Model saved to resultados/minigpt_weights.pth")
    
    # Plot losses
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    
    plot_path = '../resultados/loss_curve.png'
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

if __name__ == '__main__':
    main()
