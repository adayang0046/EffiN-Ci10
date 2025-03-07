import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_efficientnet

def train_model(epochs=10, lr=0.001):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")  # Add this line to check if MPS is being used

    trainloader, testloader = get_dataloaders()
    model = get_efficientnet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")  # Add this line to confirm training started

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(trainloader):
            print(f"Processing batch {batch_idx + 1}/{len(trainloader)}")  # Debug batch loading
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

    torch.save(model.state_dict(), "efficientnet_cifar10.pth")
    print("Model saved!")

if __name__ == "__main__":
    train_model()
