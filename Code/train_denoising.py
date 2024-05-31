import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Define the denoising autoencoder model
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)

# Define paths to the noisy and clean image folders
noisy_folder_path = '/home/pradoom/Project/Denoising/Final_train/noisy'
clean_folder_path = '/home/pradoom/Project/Denoising/Final_train/clean'

# Define data transformations (resize, convert to tensor, normalize, etc.)
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale if needed
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Create ImageFolder datasets for noisy and clean images
noisy_dataset = ImageFolder(root=noisy_folder_path, transform=transform)
clean_dataset = ImageFolder(root=clean_folder_path, transform=transform)

# Combine the datasets into one using DataLoader
combined_dataset = TensorDataset(torch.stack([img for img, _ in noisy_dataset]), torch.stack([img for img, _ in clean_dataset]))

# Create DataLoader for training
dataloader = DataLoader(combined_dataset, batch_size=2, shuffle=True)

# Initialize the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 15

for epoch in range(num_epochs):
    for batch_noisy, batch_clean in dataloader:
        # Move data to GPU
        batch_noisy, batch_clean = batch_noisy.to(device), batch_clean.to(device)

        # Forward pass
        outputs = model(batch_noisy)

        # Compute the loss
        loss = criterion(outputs, batch_clean)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'denoising_model_16a_1.pth')