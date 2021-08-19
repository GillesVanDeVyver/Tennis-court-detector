from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="..\\tennis_data")
trainer.setTrainConfig(object_names_array=["..\\tennis_court"], batch_size=16, num_experiments=100, train_from_pretrained_model="tennis_data\models\detection_model-ex-001--loss-0072.113.h5")
#trainer.setTrainConfig(object_names_array=["tennis_court"], batch_size=16, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
#download pre-trained model via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
# If you are training to detect more than 1 object, set names of objects above like object_names_array=["hololens", "google-glass", "oculus", "magic-leap"]
trainer.trainModel()
