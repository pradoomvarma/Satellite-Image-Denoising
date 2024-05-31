from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 427137840

import os

# Add multiplicative noise to an image
def add_multiplicative_noise_to_image(image_path, scale=0.5):

    """Parameters:
    - image_path: str, the path to the input image
    - scale: float, controls the intensity of the noise

    Returns:
    - noisy_image: numpy array, the image with multiplicative noise
    """
    # Open the image using Pillow
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Generate multiplicative noise
    noise = np.random.normal(loc=1.0, scale=scale, size=image_array.shape)

    # Add noise to the image
    noisy_image_array = image_array * noise

    # Clip values to ensure they stay within the valid range [0, 255]
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)

    # Convert the NumPy array back to an image
    noisy_image = Image.fromarray(noisy_image_array)

    return noisy_image


image_dir = "/home/pradoom/Documents/Data_1024/1"
all_images = os.listdir(image_dir)

noisy_images_dir = "Test_Noisy_Data_1024_1"
os.makedirs(noisy_images_dir, exist_ok=True)

for im in all_images:
    image_path = os.path.join(image_dir, im)
    noisy_image = add_multiplicative_noise_to_image(image_path)

    noisy_image_path = os.path.join(noisy_images_dir, im)
    noisy_image.save(noisy_image_path)