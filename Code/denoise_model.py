import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = 432785250

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
# model.eval()

# Define data transformation for inference
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Load and preprocess a test image
image_path = 's1a_iw_grd_vh_20220715t120413_20220715t120438_044111_0543e4_002.tiff'
image_arr = Image.open(image_path)
image_arr = np.array(image_arr)

height, width = image_arr.shape

patch_width, patch_height = 1024, 1024
counter = 0

new_image_arr = image_arr.copy()

for hg in range(0, height, patch_height):
    for wd in range(0, width, patch_width):
        cropped_im_arr = image_arr[hg:hg+patch_height, wd:wd+patch_width]
        cropped_im = Image.fromarray(cropped_im_arr.astype(np.uint8))

        input_tensor = transform(cropped_im).unsqueeze(0)
        output_patch = model(input_tensor)

        output_image = transforms.ToPILImage()(output_patch.squeeze(0))
        arr = np.array(output_image)

        try:
            new_image_arr[hg:hg+patch_height, wd:wd+patch_width] = arr
        except ValueError:
            remaining_height = min(patch_height, height - hg)
            remaining_width = min(patch_width, width - wd)
            
            new_image_arr[hg:hg+patch_height, wd:wd+patch_width] = np.resize(arr, (remaining_height, remaining_width))

# Convert output image to 8-bit integer mode (grayscale)
output_image = Image.fromarray(new_image_arr).convert('L')

# Save the denoised image as JPEG
output_image.save('denoised_image12.jpg')
