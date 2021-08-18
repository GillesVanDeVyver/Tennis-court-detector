from pgmagick import Image
from PIL import Image as pilimg
import os
import PIL
import numpy as np
from tqdm import tqdm
import gc
from augmentor import augment_images

PIL.Image.MAX_IMAGE_PIXELS = 933120000
directory = 'tennis_data\\train\images'

i=1
image_list=[]
img_locations=[]
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    print(i)
    gc.collect()
    if i>0:
        if os.path.isfile(f):
            print(f)
            img =pilimg.open(f)  # Input Image
            target_log_jpg = f
            photo_name = target_log_jpg[0:-4]
            print(photo_name)
            target_log_jpg = photo_name + "_aug.jpg"
            print(target_log_jpg)
            image1 = np.array(img, dtype=np.uint8)
            image_list.append(image1)
            img_locations.append(target_log_jpg)

            #img.write(target_log_jpg)
           # tiles = image_slicer.slice(target_log_jpg, 64, save=False)
            #pre = photo_name
           # image_slicer.save_tiles(tiles, directory='slices', format='jpg', prefix=pre)
            img=None
            tiles=None
    i=i+1
print(len(image_list))
image_as_np_arr = np.array(image_list)
images_aug = augment_images(image_as_np_arr)
print(image_as_np_arr.shape)
shape_img = images_aug.shape
print(shape_img)
for i in range(len(image_list)):
    img = pilimg.fromarray(images_aug[i], 'RGB')
    img.save(img_locations[i])
    img.show()

