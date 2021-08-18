from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="tennis_data")
trainer.setTrainConfig(object_names_array=["tennis_court"], batch_size=4, num_experiments=2, train_from_pretrained_model="pretrained-yolov3.h5")
#download pre-trained model via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
# If you are training to detect more than 1 object, set names of objects above like object_names_array=["hololens", "google-glass", "oculus", "magic-leap"]
trainer.trainModel()
