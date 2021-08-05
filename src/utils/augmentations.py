import albumentations as albu
import cv2

def aug_transforms():
    return [
        albu.VerticalFlip(p=0.7),
        albu.HorizontalFlip(p=0.7),
        albu.Rotate (limit=180, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, value=None, mask_value=None, always_apply=False, p=0.6),
        albu.ElasticTransform (alpha=10, sigma=50, alpha_affine=28,
                               interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, value=None,
                               mask_value=None, always_apply=False, approximate=False, p=0.6),
        albu.GridDistortion (num_steps=20, distort_limit=0.2, interpolation=cv2.INTER_LANCZOS4,
                             border_mode=cv2.BORDER_WRAP, value=None, mask_value=None,
                             always_apply=False, p=0.5)
    ]


transforms = albu.Compose(aug_transforms())
augFact_N = 4


def augment_image(image, mask):
    return transforms(image=image, mask=mask)
