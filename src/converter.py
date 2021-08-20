"""
This script reads pictures in the given directory (in this case .jp2 files)
It converts the pictures to .jpg files and saves them in the given target folder.
Next it splits the images into 64 equal squares and saves these in the given target folder

@param directory: directory to read original images from
@param target_for_jpg: target directory to write jpg files to
@param target_for_slices: target directory to write slices to
"""

directory = '..\VLAANDEREN_WINTER_2019'
target_for_jpg='..\data as jpeg\\'
target_for_slices='..\slices'






from pgmagick import Image
import os
import PIL
import image_slicer
from tqdm import tqdm
import gc




PIL.Image.MAX_IMAGE_PIXELS = 933120000
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    gc.collect()
    if os.path.isfile(f):
        print("reading" + str(f))
        img = Image(f)
        target_loc_jpg = f
        photo_name = target_loc_jpg[len(directory)+1:-4]
        print(photo_name)
        target_loc_jpg = target_for_jpg + photo_name + ".jpg"
        print('writing' + str(target_loc_jpg))
        img.write(target_loc_jpg)
        print(target_loc_jpg)
        tiles = image_slicer.slice(target_loc_jpg, 64, save=False)
        pre = photo_name
        print('writing slices at' + str(target_for_slices))
        image_slicer.save_tiles(tiles, directory=target_for_slices, format='jpg', prefix=pre)
        img=None #clear memory
        tiles=None






