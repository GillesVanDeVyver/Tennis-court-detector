import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np


ia.seed(1)

def augment_images(image_as_np_arr):
    seq = iaa.Sequential([
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.3))
        ),
        iaa.Sometimes(
            0.5,
            iaa.LinearContrast((0.7, 1.3)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.015 * 255), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Multiply((0.75, 1.15), per_channel=0.2),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Add((-7.5, 7.5), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Multiply((0.85, 1.15), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Grayscale(alpha=(0.0, 0.15)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.ElasticTransformation(alpha=(0, 0.75), sigma=0.25)
        ),
    ], random_order=True)

    images_aug = seq(images=image_as_np_arr)
    return images_aug


#image1 = np.array(Image.open('tennis_data\\train\images\MWRGBMRVL_K131n_08_02.jpg'),dtype=np.uint8)
#image2 = np.array(Image.open('tennis_data\\train\images\MWRGBMRVL_K131n_07_02.jpg'),dtype=np.uint8)

#image_as_np_arr=np.array((image1,image2))
#print(image_as_np_arr.shape)

#augment_images(image_as_np_arr)
