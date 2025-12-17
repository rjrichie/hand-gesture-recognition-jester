import random
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

def augment_single_image_to_jpg(
    img: Image.Image,
    resize_size=(144, 144),
    crop_size=(128, 128),
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    hue_range=(-0.1, 0.1),  # should already be small
    rotation_range=(-5, 5),
):
    # 1. Resize
    img = TF.resize(img, resize_size)

    # 2. Random crop
    i, j, h, w = T.RandomCrop.get_params(img, output_size=crop_size)
    img = TF.crop(img, i, j, h, w)

    # 3. Color jitter (manual params)
    brightness_factor = random.uniform(*brightness_range)
    contrast_factor   = random.uniform(*contrast_range)
    saturation_factor = random.uniform(*saturation_range)
    hue_factor        = random.uniform(*hue_range)

    # 🔒 clamp hue to valid range [-0.5, 0.5]
    hue_factor = max(-0.5, min(0.5, hue_factor))

    img = TF.adjust_brightness(img, brightness_factor)
    img = TF.adjust_contrast(img,   contrast_factor)
    img = TF.adjust_saturation(img, saturation_factor)
    img = TF.adjust_hue(img,        hue_factor)

    # 4. Rotation
    angle = random.uniform(*rotation_range)
    img = TF.rotate(img, angle)

    # 5. Return PIL so you can save as JPG
    return img

from PIL import Image

img = Image.open("input.jpg").convert("RGB")
aug = augment_single_image_to_jpg(img)
aug.save("output_aug.jpg", quality=95)
