import albumentations.core.transforms_interface
import logging
import albumentations as A
from albumentations import GaussianBlur, CoarseDropout # <-- Import thêm CoarseDropout
from albumentations.augmentations import RandomBrightnessContrast

from donkeycar.config import Config

logger = logging.getLogger(__name__)

class ImageAugmentation:
    def __init__(self, cfg, key, prob=0.5):
        aug_list = getattr(cfg, key, [])
        # Thay đổi xác suất (prob) để augmentation không áp dụng lên 100% ảnh
        # Để xe vẫn học được cả trường hợp đường đẹp.
        augmentations = [ImageAugmentation.create(a, cfg, prob)
                         for a in aug_list]
        self.augmentations = A.Compose(augmentations)

    @classmethod
    def create(cls, aug_type: str, config: Config, prob) -> \
            albumentations.core.transforms_interface.BasicTransform:
        """ Augmentation factory. """

        if aug_type == 'BRIGHTNESS':
            b_limit = getattr(config, 'AUG_BRIGHTNESS_RANGE', 0.2)
            logger.info(f'Creating augmentation {aug_type} {b_limit}')
            return RandomBrightnessContrast(brightness_limit=b_limit,
                                            contrast_limit=b_limit,
                                            p=prob)

        elif aug_type == 'BLUR':
            b_range = getattr(config, 'AUG_BLUR_RANGE', 3)
            logger.info(f'Creating augmentation {aug_type} {b_range}')
            return GaussianBlur(sigma_limit=b_range, blur_limit=(13, 13),
                                p=prob)
        
        elif aug_type == 'CUTOUT':
            # Lấy thông số từ myconfig.py
            num_holes = getattr(config, 'AUG_CUTOUT_HOLES', 8)
            max_h = getattr(config, 'AUG_CUTOUT_SIZE', 20)
            max_w = getattr(config, 'AUG_CUTOUT_SIZE', 20)
            # Fill value = 255 (Màu trắng) để giả lập việc vạch đen bị che bởi giấy trắng
            fill_val = getattr(config, 'AUG_CUTOUT_FILL', 255) 
            
            logger.info(f'Creating augmentation {aug_type}: Holes={num_holes}, Size={max_h}, Fill={fill_val}')
            
            # Sử dụng CoarseDropout của Albumentations
            return A.CoarseDropout(
                max_holes=num_holes,
                max_height=max_h,
                max_width=max_w,
                min_holes=1,
                min_height=4,
                min_width=4,
                fill_value=fill_val, # Quan trọng: Màu của vùng bị che
                p=prob
            )
        # -------------------------

    # Parts interface
    def run(self, img_arr):
        if len(self.augmentations) == 0:
            return img_arr
        aug_img_arr = self.augmentations(image=img_arr)["image"]
        return aug_img_arr