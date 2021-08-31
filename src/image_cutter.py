"""
This script was added as an attempt to improve accuracy.
The hypothesis was the poor detection quality was due to the images being too big.

The script takes images and annotation in the given directories, cuts them into slices
and writes them to target directory. Slices without annotations are discarded

@param directory: directory to read original images from
@param xml_dir: directory to read original annoations from
@param target_for_slices: target directory to write images to
@param target_for_slice_annotations: target directory to write annotations to
"""

xml_dir='..\\tennis_data\\validation\\annotations'
target_for_slices='..\slices'
target_for_slice_annotations='..\slice_annotations'

directory='..\\tennis_data\\validation\\images'





from pgmagick import Image
import image_slicer


import copy
import xml.etree.ElementTree as ET
import os
import PIL
from tqdm import tqdm
import gc





PIL.Image.MAX_IMAGE_PIXELS = 933120000
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    gc.collect()
    if os.path.isfile(f):
        print("reading" + str(f))
        img = Image(f)
        photo_name = f[len(directory)+1:-4]
        tiles = image_slicer.slice(f, 16, save=False)
        pre = photo_name
        print(photo_name)
        print('writing slices at' + str(target_for_slices))
        image_slicer.save_tiles(tiles, directory=target_for_slices, format='jpg', prefix=pre)
        tiles=None #clear memory



count = 0
count_orig=0
for filename in tqdm(os.listdir(xml_dir)):
    f = os.path.join(xml_dir, filename)
    gc.collect()
    if os.path.isfile(f):
        print(f)
        source_xml_loc=f
        tree = ET.parse(f)
        root = tree.getroot()
        new_tree= copy.deepcopy(tree)
        new_root= new_tree.getroot()
        adjustment=0
        objects=[]
        for i in range(len(root)):
            if root[i].tag=="object":
                count_orig=count_orig+1
                objects.append(root[i])
                new_root.remove(new_root[i-adjustment])
                adjustment=adjustment+1
        img_name=f[len(xml_dir)+1:-4]
        target_loc_common=target_for_slice_annotations+'\\'+img_name

        y_bounds_min=0
        y_bounds_max=625
        for y_ind in range(4):
            x_bounds_min = 0
            x_bounds_max = 1000
            for x_ind in range(4):
                new_tree_to_write=copy.deepcopy(new_tree)
                new_root_to_write= new_tree_to_write.getroot()
                to_write=False
                for obj in objects:
                    x_min = int(obj[4][0].text)
                    y_min = int(obj[4][1].text)
                    x_max = int(obj[4][2].text)
                    y_max = int(obj[4][3].text)
                    if x_min > x_bounds_min and x_max <x_bounds_max and\
                        y_min > y_bounds_min and y_max < y_bounds_max:
                        obj_to_add=obj
                        obj_to_add[4][0].text = str(int(obj_to_add[4][0].text) - x_bounds_min)
                        obj_to_add[4][1].text = str(int(obj_to_add[4][1].text) - y_bounds_min)
                        obj_to_add[4][2].text = str(int(obj_to_add[4][2].text) - x_bounds_min)
                        obj_to_add[4][3].text = str(int(obj_to_add[4][3].text) - y_bounds_min)
                        to_write=True
                        count=count+1
                        new_root_to_write.append(obj_to_add)
                if to_write:
                    slice_name="_0"+str(y_ind+1)+"_0"+str(x_ind+1)
                    new_root_to_write[1].text = img_name +slice_name+".jpg" #adjust filename
                    target_loc=target_loc_common+slice_name+".xml"
                    new_tree_to_write.write(target_loc, encoding='utf-8')
                x_bounds_max = x_bounds_max + 1000
                x_bounds_min = x_bounds_min + 1000
            y_bounds_max = y_bounds_max + 625
            y_bounds_min = y_bounds_min + 625
print(count)
print(count_orig)

# remove slices without annotations
for filename in tqdm(os.listdir(target_for_slices)):
    f = os.path.join(target_for_slices, filename)
    gc.collect()
    if os.path.isfile(f):
        try:
            xml_loc='..\slice_annotations\\'+filename[:-3]+'xml'
            f2 = open(xml_loc)
        except IOError:
            os.remove(f)





