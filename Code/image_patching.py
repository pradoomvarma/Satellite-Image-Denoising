import os
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = 432785250

def patch_image(image_path, dest_dir_path):
    os.makedirs(dest_dir_path, exist_ok=True)
    os.makedirs(os.path.join(dest_dir_path, image_path.split("/")[-1].split(".")[0]), exist_ok=True)

    image_arr = Image.open(image_path)
    image_arr = np.array(image_arr)
    height, width = image_arr.shape

    patch_width, patch_height = 1024, 1024
    counter = 0

    for hg in range(0, height, patch_height):
        for wd in range(0, width, patch_width):
            cropped_im_arr = image_arr[hg:hg+patch_height, wd:wd+patch_width]
            cropped_im = Image.fromarray(cropped_im_arr.astype(np.uint8))

            dest_im_path = os.path.join(dest_dir_path, image_path.split("/")[-1].split(".")[0], image_path.split("/")[-1].split(".")[0] + "_" + str(counter) + ".jpeg")
            cropped_im.save(dest_im_path)

            counter += 1

clean_image_dir = "/home/pradoom/Project/Denoising/Data"
# noisy_image_dir = "./dataset/noisy_images"

for dir in [clean_image_dir]:
    images = os.listdir(dir)
    patched_images_path = dir.replace(dir.split('/')[-1], "patche_"+dir.split('/')[-1])

    for im in images:
        im_path = os.path.join(dir, im)

        patch_image(im_path, patched_images_path)

    print(f"Done all images for dir {dir}")

print("Done")