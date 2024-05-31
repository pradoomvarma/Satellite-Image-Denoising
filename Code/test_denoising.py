import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import math

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

# Load the pre-trained model
model = DenoisingAutoencoder()
model.load_state_dict(torch.load('denoising_model_16a_1.pth'))
model.eval()

# Define data transformation for inference
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Load and preprocess the original and noisy images
original_image_path = '/home/pradoom/Project/Denoising/Final_test/Clean/clean/3_291.jpeg'
noisy_image_path = '/home/pradoom/Project/Denoising/Final_test/Noisy/noisy/3_291.jpeg'

original_image = Image.open(original_image_path).convert('L')
noisy_image = Image.open(noisy_image_path).convert('L')

original_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension
noisy_tensor = transform(noisy_image).unsqueeze(0)  # Add batch dimension

# Calculate PSNR before denoising
def calculate_psnr(image1, image2):
    mse = ((image1 - image2) ** 2).mean()
    max_pixel_value = 255
    psnr = 20 * math.log10(max_pixel_value) - 10 * math.log10(mse)
    return psnr

psnr_before_denoising = calculate_psnr(original_tensor.squeeze(0).numpy() * 255, noisy_tensor.squeeze(0).numpy() * 255)

# Perform denoising
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

# Calculate PSNR after denoising
psnr_after_denoising = calculate_psnr(original_tensor.squeeze(0).numpy() * 255, denoised_tensor.squeeze(0).numpy() * 255)

print(f"PSNR before denoising: {psnr_before_denoising:.2f} dB")
print(f"PSNR after denoising: {psnr_after_denoising:.2f} dB")

# Convert the denoised tensor to a PIL image
denoised_image = transforms.ToPILImage()(denoised_tensor.squeeze(0))

# Save the denoised image
denoised_image.save('denoised_image.jpeg')
