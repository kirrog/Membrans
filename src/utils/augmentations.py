import sys
import albumentations as albu
import cv2
import numpy as np
import tensorflow as tf

from keras import layers

# data_augmentation = tf.keras.Sequential([
#     layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
#     layers.experimental.preprocessing.RandomRotation(0.2),
# ])


# def aug_image_by_keras(img):
#     return data_augmentation(img)


def aug_transforms():
    return [
        albu.VerticalFlip(),
        albu.HorizontalFlip(),
        albu.Rotate(limit=180, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, always_apply=False,
                    p=0.6),
        albu.ElasticTransform(alpha=10, sigma=50, alpha_affine=28,
                              interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP,
                              always_apply=False, approximate=False, p=0.6),
        albu.GridDistortion(num_steps=20, distort_limit=0.2, interpolation=cv2.INTER_LANCZOS4,
                            border_mode=cv2.BORDER_WRAP,
                            always_apply=False, p=0.5)
    ]


transforms = albu.Compose(aug_transforms())
img_x = 512
img_y = 512


def augment_dataset(images, masks):
    augmentated_images = np.zeros((len(images), img_x, img_y, 1), dtype=np.float16)
    augmentated_masks = np.zeros((len(masks), img_x, img_y, 1), dtype=np.uint8)
    for i in range(len(images)):
        a = transforms(image=np.float32(images[i]), mask=np.float32(masks[i]))
        augmentated_images[i] = a['image']
        augmentated_masks[i] = a['mask']
        sys.stdout.write("\rImage %i augmentated" % i)
    return augmentated_images, augmentated_masks


def augment_image(image, mask):
    res = transforms(image=image, mask=mask)
    return res["image"], res["mask"]
