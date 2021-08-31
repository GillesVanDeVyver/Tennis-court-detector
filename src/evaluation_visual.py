"""
Applies the given model to the given picture

@param img_loc: path to image
@param model_loc: path to model
@param config_loc: path to json config

@param output_name: relative output path and name
"""
img_loc="..\\tennis_data\\validation\images\MWRGBMRVL_K048n_07_07_04_03.jpg"
model_loc="..\\tennis_data\models\detection_model-ex-001--loss-0046.868.h5"
config_loc="..\\tennis_data\json\detection_config.json"

output_name='detected_image.jpg'

verbose=False








from imageai.Detection.Custom import CustomObjectDetection
import cv2
from PIL import Image

image_array = cv2.imread(img_loc)
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_loc)
detector.setJsonPath(config_loc)
detector.loadModel()

detected_image, detections = detector.detectObjectsFromImage(input_image=image_array, input_type="array", output_type="array", minimum_percentage_probability=20)
if verbose:
    for eachObject in detections:
        print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
    print(len(detections))
cv2.imwrite(output_name, detected_image)
cv2.waitKey()
cv2.destroyAllWindows()
