# scripts/train.py

import sys
import os
import copy  # Import for deep copy of model weights

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now try to import the Generator
try:
    from models.generator import ConvGenerator  # Import your updated generator model
    print("Import successful!")
except ModuleNotFoundError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import for learning rate scheduler

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust normalization as per your data
])

# Function to verify dataset structure
def verify_dataset_structure(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist.")
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    if not subdirs:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    print(f"Found class folders in {directory}: {subdirs}")

# Verify dataset structure
try:
    verify_dataset_structure('data/processed_train')
    verify_dataset_structure('data/processed_validation')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# Load datasets using ImageFolder
train_dataset = ImageFolder('data/processed_train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Use the best batch size
validation_dataset = ImageFolder('data/processed_validation', transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)  # Use the best batch size

# Initialize generator model
generator = ConvGenerator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))  # Use the best learning rate
criterion = nn.MSELoss()  # Example loss function, adjust as per your task

# Initialize the learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
num_epochs = 100  # Increase number of epochs


for epoch in range(num_epochs):
    generator.train()  # Set model to training mode
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)  # Move data to GPU
        noise = torch.randn(data.size(0), 100).to(device)  # Generate noise input for the generator
        optimizer.zero_grad()  # Zero the gradients
        output = generator(noise)  # Forward pass
        loss = criterion(output, data)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Print training statistics (optional)
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Validation: Evaluate model performance on validation set after each epoch
    generator.eval()  # Set model to evaluation mode
    val_loss_total = 0.0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(validation_loader):
            data = data.to(device)  # Move data to GPU
            noise = torch.randn(data.size(0), 100).to(device)  # Generate noise input for the generator
            output = generator(noise)
            val_loss = criterion(output, data)
            val_loss_total += val_loss.item()

    # Calculate average validation loss
    avg_val_loss = val_loss_total / len(validation_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

    # Adjust learning rate based on validation loss
    scheduler.step(avg_val_loss)

    # Save checkpoint (optional)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': loss.item(),
        'val_loss': avg_val_loss
    }
    torch.save(checkpoint, f'models/checkpoint_epoch_{epoch+1}.pth')

# Save final trained model weights
torch.save(generator.state_dict(), 'models/generator_final.pth')
