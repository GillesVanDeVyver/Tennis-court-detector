from imageai.Detection.Custom import CustomObjectDetection
import cv2
from PIL import Image

image_array = cv2.imread("tennis_data\\validation\images\MWRGBMRVL_K048n_07_07.jpg")
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("tennis_data\models\detection_model-ex-002--loss-0057.188.h5")
detector.setJsonPath("tennis_data\json\detection_config.json")
detector.loadModel()

detected_image, detections = detector.detectObjectsFromImage(input_image=image_array, input_type="array", output_type="array", minimum_percentage_probability=15)
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])

#img = Image.fromarray(detected_image)
#img.save("detected_image.jpg")
print(len(detections))
cv2.imshow("Main Image", detected_image)
cv2.imwrite('detected_image2.jpg', detected_image)

cv2.waitKey()
cv2.destroyAllWindows()
