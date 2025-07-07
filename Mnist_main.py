import numpy as np
import os
from tqdm import tqdm, trange

import torch

from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from Models.VIT_MNIST import MyViT

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from Utils.Mnist_Utils import convert_to_pt, save_model_ckpt
from datetime import datetime
np.random.seed(0)
torch.manual_seed(0)
from time import time

def main():
    # Loading data
    transform = ToTensor()
    mnist_root = r"E:\Vision_Transformers\Mnist_Data"

    # Example usage:
    convert_to_pt(os.path.join(mnist_root, "MNIST", "raw"), os.path.join(mnist_root, "MNIST", "processed"))

    train_set = MNIST(root=mnist_root, train=True, download=False, transform=transform)
    test_set = MNIST(root=mnist_root, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=2048, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=2048, num_workers=4, pin_memory=True, persistent_workers=True)

    timestamp = datetime.now().strftime('%d_%b_%Y___%H-%M-%S_%p')
    save_model_dir = os.path.join("Mnist_Data", "saved_models", timestamp)
    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = MyViT((1, 28, 28), n_patches=4, n_blocks=4, hidden_d=16, n_heads=4, out_d=10).to(device)
    N_EPOCHS = 50
    LR = 0.005
    step = 0
    best_test_acc = 80.0
    # Training loop
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    print("Using CUDA:", torch.cuda.is_available())
    print("Model on device:", next(model.parameters()).device)
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

        # Test loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(test_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(test_loader)
                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
            print(f"Test loss: {test_loss:.2f}")
            accuracy = correct / total * 100
            if accuracy > best_test_acc:
                best_test_acc = accuracy
                save_model_ckpt(save_model_dir, step, model, accuracy)
            print(f"Test accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
