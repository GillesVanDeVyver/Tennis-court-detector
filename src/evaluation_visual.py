from imageai.Detection.Custom import CustomObjectDetection
import cv2
from PIL import Image

#image_array = cv2.imread("tennis_data\\validation\images\MWRGBMRVL_K048n_07_07.jpg")
image_array = cv2.imread("..\\tennis_data\\train\images\MWRGBMRVL_K018z_04_05.jpg")
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("..\\tennis_data\models\detection_model-ex-001--loss-0072.113.h5")
detector.setJsonPath("..\\tennis_data\json\detection_config.json")
detector.loadModel()

detected_image, detections = detector.detectObjectsFromImage(input_image=image_array, input_type="array", output_type="array", minimum_percentage_probability=30)
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])

#img = Image.fromarray(detected_image)
#img.save("detected_image.jpg")
print(len(detections))
#cv2.imshow("Main Image", detected_image)
#cv2.imwrite('detected_image2.jpg', detected_image)
cv2.imwrite('detected_image2_train.jpg', detected_image)

cv2.waitKey()
cv2.destroyAllWindows()
