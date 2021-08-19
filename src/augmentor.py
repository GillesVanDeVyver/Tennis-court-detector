import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
from PIL import Image as pilimg
import os
import PIL
import numpy as np
from tqdm import tqdm
import gc

ia.seed(1)

def augment_images(image_as_np_arr):
    seq = iaa.Sequential([
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.2))
        ),
        iaa.Sometimes(
            0.5,
            iaa.LinearContrast((0.8, 1.2)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Multiply((0.7, 1.3), per_channel=0.2),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Add((-8.5, 8.5), per_channel=0.5),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Grayscale(alpha=(0.0, 0.15)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.75)),
        ),
        iaa.Sometimes(
            0.5,
            iaa.GammaContrast(1.5),
        ),


    ], random_order=True)

    images_aug = seq(images=image_as_np_arr)
    return images_aug


nb_of_augmentations=10

PIL.Image.MAX_IMAGE_PIXELS = 933120000
directory = '..\\tennis_data\\train\images'

image_list=[]
img_locations=[]
print("reading images...")
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    gc.collect()
    if os.path.isfile(f):
        if not "aug" in f:
            img =pilimg.open(f)  # Input Image
            target_log_jpg = f
            photo_name = target_log_jpg[0:-4]
            target_log_jpg = photo_name + "_aug.jpg"
            image1 = np.array(img, dtype=np.uint8)
            image_list.append(image1)
            img_locations.append(target_log_jpg)
        img=None
print("read " +str(len(image_list)) + " images")
for i in tqdm(range(nb_of_augmentations)):
    print("augmenting images...")
    image_as_np_arr = np.array(image_list)
    images_aug = augment_images(image_as_np_arr)
    shape_img = images_aug.shape
    print("saving augmented images...")
    for j in range(len(image_list)):
        img = pilimg.fromarray(images_aug[j], 'RGB')
        img.save(img_locations[j][0:-4]+str(i)+".jpg")
