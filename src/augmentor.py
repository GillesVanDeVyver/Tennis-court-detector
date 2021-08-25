"""
This script augments the data in the given directory.
This directory needs to have a subfolder images with .jpg files
and a subfolder annotations with .xml files

The augmentations are:
type I annotations: blur, sharpen, color adjustment..
type II annotations: horizontal and vertical flips

@param directory: directory with data
@param nb_of_typeI_augmentations: the number of iterations for every imqge for type I augmentations
"""

directory = '..\\tennis_data\\train'
nb_of_typeI_augmentations=8


img_size=(4000,2500,3)



import imgaug as ia
from PIL import Image
from shutil import copyfile
from lxml import etree
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
from PIL import Image
from PIL import Image as pilimg
import os
import PIL
import numpy as np
from tqdm import tqdm
import gc

########Helper functions########

# fix for issue:
# annotation tool labelImg produces xml without extension in filename
# annotation tool labelImg produces xml without correct size label
def fix_xml(xml_path,img_size):
    mytree = ET.parse(xml_path)
    myroot = mytree.getroot()
    img_name_attr=myroot.find('filename')
    if ".jpg" not in img_name_attr.text:
        img_name_attr.text = str(img_name_attr.text + ".jpg")
    size_attr=myroot.find('size')
    width_attr=size_attr.find("width")
    width_attr.text=str(img_size[0])
    height_attr=size_attr.find("height")
    height_attr.text=str(img_size[1])
    mytree.write(xml_path)

class CreateAnnotations:
    # -----Initialization
    def __init__(self, flodername, filename):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = flodername

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "path")
        child3.text = filename

        child4 = etree.SubElement(self.root, "source")

        child5 = etree.SubElement(child4, "database")
        child5.text = "Unknown"

    # -----Set size
    def set_size(self, imgshape):
        (height, witdh, channel) = imgshape
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    # -----Save the file
    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)
ia.seed(1)
PIL.Image.MAX_IMAGE_PIXELS = 933120000
# type I annotations: blur, sharpen, color adjustment..

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

def save_file(bbs_aug, target_location,image_aug):
    foldername = ""

    anno = CreateAnnotations(foldername, target_location[25+3:])
    anno.set_size(image_aug.shape)
    for index, bb in enumerate(bbs_aug):
        xmin = int(bb.x1)
        ymin = int(bb.y1)
        xmax = int(bb.x2)
        ymax = int(bb.y2)
        label = str(bb.label)
        anno.add_pic_attr(label, xmin, ymin, xmax, ymax)
    anno.savefile(get_tqrget_xml_loc(target_location))
    imageio.imsave(target_location, image_aug)

def get_xml_loc(f):
    offset=4
    if 'aug' in f:
        offset=offset+5
    if 'hor' in f:
        offset=offset+4
    if 'ver' in f:
        offset=offset+4
    return anno_dir + '\\' + f[len(img_dir):-offset] + '.xml'

def get_tqrget_xml_loc(f):
    return anno_dir + '\\' + f[len(img_dir):-3] + 'xml'

########start of preprocessing instructions########

img_dir=directory+"\images"
anno_dir=directory+"\\annotations"

# fix XML files


for filename in os.listdir(img_dir):
    f = os.path.join(img_dir, filename)
    gc.collect()
    if os.path.isfile(f):
        xml_path = get_xml_loc(f)
        fix_xml(xml_path,img_size)


# execute type I augmentations

image_list=[]
img_locations=[]
print("reading images...")
for filename in os.listdir(img_dir):
    f = os.path.join(img_dir, filename)
    gc.collect()
    if os.path.isfile(f):
        if not "aug" in f and not "hor" in f and not "ver" in f:
            img =pilimg.open(f)
            target_log_jpg = f
            photo_name = target_log_jpg[0:-4]
            target_log_jpg = photo_name + "_aug.jpg"
            image1 = np.array(img, dtype=np.uint8)
            image_list.append(image1)
            img_locations.append(target_log_jpg)
        img=None
print("read " +str(len(image_list)) + " images")
for i in tqdm(range(nb_of_typeI_augmentations)):
    print("augmenting images...")
    image_as_np_arr = np.array(image_list)
    images_aug = augment_images(image_as_np_arr)
    shape_img = images_aug.shape
    print("saving augmented images...")
    for j in range(len(image_list)):
        img = pilimg.fromarray(images_aug[j], 'RGB')
        img.save(img_locations[j][0:-4]+str(i)+".jpg")


# for type I augmentations the annotations remain the same => copy xml files
# NOTE: this needs to be executed before flipping

for filename in tqdm(os.listdir(img_dir)):
    f = os.path.join(img_dir, filename)
    gc.collect()
    if os.path.isfile(f):
        if "aug" in f and not 'hor' in f and not 'ver' in f:
            target_xml_loc=get_tqrget_xml_loc(f)
            source_xml_loc=xml_loc=get_xml_loc(f)
            img_name=f[len(directory)+1:]
            copyfile(source_xml_loc, target_xml_loc)
            root = etree.Element("annotation")
            child2 = etree.SubElement(root, "filename")
            child2.text = img_name

# type II annotations: horizontal and vertical flips

print("flipping images...")
for filename in tqdm(os.listdir(img_dir)):
    f = os.path.join(img_dir, filename)
    gc.collect()
    if os.path.isfile(f):
        if not "hor" in f and not "ver" in f and not 'xml' in f:
            jpg_loc=f
            xml_loc=get_xml_loc(f)
            target_loc=jpg_loc[0:-4]
            geo_augment(xml_loc, jpg_loc, target_loc)
