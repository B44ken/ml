import torch
import torchvision
import json
import os
from torchvision import transforms

def download_and_save_mnist():
    # Create dataset directory if it doesn't exist
    os.makedirs('dataset', exist_ok=True)

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])

    # Download MNIST data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Function to save data to JSON file
    def save_to_json(dataset, prefix, num_samples):
        for i in range(num_samples):
            image, label = dataset[i]
            data = {
                "X": image.numpy().flatten().tolist(),
                "y": int(label)
            }
            filename = f"dataset/{prefix}{i+1}.json"
            with open(filename, 'w') as f:
                json.dump(data, f)

    # Save training data
    save_to_json(train_dataset, "train", 600)

    # Save test data
    save_to_json(test_dataset, "test", 100)

    print("MNIST data has been downloaded and saved to JSON files.")

if __name__ == "__main__":
    download_and_save_mnist()

