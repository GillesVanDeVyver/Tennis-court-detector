from pgmagick import Image
import os
import PIL
#from PIL import Image
import image_slicer
from tqdm import tqdm
import gc


PIL.Image.MAX_IMAGE_PIXELS = 933120000
directory = '..\VLAANDEREN_WINTER_2019'

i=1
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    print(i)
    gc.collect()
    if i>0:
        if os.path.isfile(f):
            print(f)
            img = Image(f)  # Input Image
            target_log_jpg = f
            photo_name = target_log_jpg[24:-4]
            print(photo_name)
            target_log_jpg = 'data as jpeg\O' + photo_name + ".jpg"
            print(target_log_jpg)
            img.write(target_log_jpg)
            tiles = image_slicer.slice(target_log_jpg, 64, save=False)
            pre = photo_name
            image_slicer.save_tiles(tiles, directory='slices', format='jpg', prefix=pre)
            img=None
            tiles=None
    i=i+1





