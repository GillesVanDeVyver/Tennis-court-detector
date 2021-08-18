from bs4 import BeautifulSoup
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from annotation_converter import CreateAnnotations
import imgaug.augmenters as iaa
from PIL import Image
from PIL import Image as pilimg
import os
import PIL
import numpy as np
from tqdm import tqdm
import gc












def save_file(bbs_aug, target_location,image_aug):
    foldername = ""

    anno = CreateAnnotations(foldername, target_location)
    anno.set_size(image_aug.shape)
    for index, bb in enumerate(bbs_aug):
        xmin = int(bb.x1)
        ymin = int(bb.y1)
        xmax = int(bb.x2)
        ymax = int(bb.y2)
        label = str(bb.label)
        anno.add_pic_attr(label, xmin, ymin, xmax, ymax)
    anno.savefile("{}.xml".format(target_location.split(".")[0]))
    imageio.imsave(target_location, image_aug)

def geo_augment(xml_loc,jpg_loc,target_loc):

    soup = BeautifulSoup(open(xml_loc), "lxml")
    image = imageio.imread(jpg_loc)
    target_location = target_loc
    bbsOnImg=[]
    for objects in soup.find_all(name="object"):
        object_name = str(objects.find(name="name").string)
        xmin = int(objects.xmin.string)
        ymin = int(objects.ymin.string)
        xmax = int(objects.xmax.string)
        ymax = int(objects.ymax.string)
        bbsOnImg.append(BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax,label=object_name))
    bbs = BoundingBoxesOnImage( bbsOnImg,shape=image.shape)



    seq_flip_hor = iaa.Sequential([
        iaa.Fliplr(1),
    ])
    image_aug_hor, bbs_aug_hor = seq_flip_hor(image=image, bounding_boxes=bbs)
    name_hor=target_location+"_hor" + ".jpg"
    save_file(bbs_aug_hor,name_hor,image_aug_hor)


    seq_flip_ver = iaa.Sequential([
        iaa.Flipud(1),
    ])
    seq_flip_ver, bbs_aug_ver = seq_flip_ver(image=image, bounding_boxes=bbs)
    name_ver=target_location+"_ver" + ".jpg"
    save_file(bbs_aug_ver,name_ver,seq_flip_ver)



    seq_flip_hor_ver = iaa.Sequential([
        iaa.Fliplr(1),
        iaa.Flipud(1),
    ])
    seq_flip_hor_ver, bbs_aug_hor_ver = seq_flip_hor_ver(image=image, bounding_boxes=bbs)
    name_hor_ver=target_location+"_hor_ver" + ".jpg"
    save_file(bbs_aug_hor_ver,name_hor_ver,seq_flip_hor_ver)







PIL.Image.MAX_IMAGE_PIXELS = 933120000
directory = 'tennis_data\\train\images'

print("flipping images...")
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    gc.collect()
    if os.path.isfile(f):
        if not "hor" in f and not "ver" in f and not 'xml' in f:
            print(f)
            jpg_loc=f
            if 'aug' in f:
                xml_loc = f[0:18] + 'annotations\\' + f[-30:-3] + 'xml'
            else:
                xml_loc=f[0:18]+'annotations\\'+f[-26:-3]+'xml'
            target_loc=jpg_loc[0:-4]
            geo_augment(xml_loc, jpg_loc, target_loc)
        img=None




