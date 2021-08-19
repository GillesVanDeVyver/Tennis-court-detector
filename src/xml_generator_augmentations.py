from shutil import copyfile
from PIL import Image
import os
import PIL
from tqdm import tqdm
import gc
from lxml import etree

# this needs to be executed before flipping

PIL.Image.MAX_IMAGE_PIXELS = 933120000
directory = '..\\tennis_data\\train\images'


print("flipping images...")
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    gc.collect()
    if os.path.isfile(f):
        if "aug" in f and not 'hor' in f and not 'ver' in f:
            target_xml_loc=f[0:18]+'annotations\\'+f[-30:-3]+'xml'
            source_xml_loc=f[0:18]+'annotations\\'+f[-30:-9]+'.xml'
            copyfile(source_xml_loc, target_xml_loc)
            print(f[-30:])

            root = etree.Element("annotation")
            child2 = etree.SubElement(root, "filename")
            child2.text = f[-30:]



        img=None



