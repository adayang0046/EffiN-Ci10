import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import get_efficientnet

def evaluate_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # for mac

    # load the test data
    _, testloader = get_dataloaders(batch_size=64)

    # load the trained model
    model = get_efficientnet(num_classes=10)
    model.load_state_dict(torch.load("efficientnet_cifar10.pth", map_location=device))
    model.to(device)
    model.eval()  # set model to eval

    correct = 0
    total = 0

    with torch.no_grad():  # gradients
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()
