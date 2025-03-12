# transfer_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


# Configuration

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # Adjust based on your dataset
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Define Data Transforms

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Load Dataset

def load_dataset(data_dir):
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Loaded {len(train_dataset)} training and {len(val_dataset)} validation images.")

    return train_loader, val_loader


# Load Pretrained Model

def load_model(model_name, num_classes, fine_tune=True):
    if model_name == 'resnet':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Invalid model. Choose from 'resnet', 'vgg', 'efficientnet'")

    if fine_tune:
        # Unfreeze model for full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Freeze feature extractor, only fine-tune classifier head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    model = model.to(DEVICE)
    return model


# Train Model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        print(f"\n[INFO] Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_description(f"Train Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_loss_list.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        val_loss_list.append(val_loss)

        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}.pth")
            print("[INFO] Best model saved!")

    print(f"\n[INFO] Training complete. Best Validation Accuracy: {best_acc:.2f}%")

    # Plot losses
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Main Function

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'vgg', 'efficientnet'])
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the whole model')
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    # Load dataset
    train_loader, val_loader = load_dataset(args.data_dir)

    # Load model
    model = load_model(args.model, NUM_CLASSES, fine_tune=args.fine_tune)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs)

if __name__ == '__main__':
    main()
