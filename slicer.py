import PIL
from PIL import Image
import image_slicer
import time
PIL.Image.MAX_IMAGE_PIXELS = 933120000

start_time = time.time()

image_slicer.slice('data as jpeg/1.jpeg', 64)

end_time = time.time()
elapsed=end_time-start_time
print(elapsed)