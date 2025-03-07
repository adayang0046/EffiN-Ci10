import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for EfficientNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def check_dataset():
    trainloader, testloader = get_dataloaders(batch_size=4)  # Load small batch for testing

    # Get a batch of images and labels
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Print dataset info
    print("Dataset successfully loaded!")
    print(f"Train dataset size: {len(trainloader.dataset)} images")
    print(f"Test dataset size: {len(testloader.dataset)} images")
    print(f"Batch size: {images.shape}")  # Expected: [4, 3, 224, 224]

    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Function to denormalize and display images
    def imshow(img):
        img = img.numpy().transpose((1, 2, 0))  # Convert from tensor to NumPy
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
        img = np.clip(img, 0, 1)  # Clip values between 0 and 1
        plt.imshow(img)
        plt.axis('off')

    # Show images
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))
    for i in range(4):
        ax = axes[i]
        imshow(images[i])
        ax.set_title(classes[labels[i].item()])
    plt.show()

# Run this function to check dataset loading
if __name__ == "__main__":
    check_dataset()
