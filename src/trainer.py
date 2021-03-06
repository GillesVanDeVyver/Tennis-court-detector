"""
This script continues training from a given pretrained model on data in a given folder.
This directory needs to have a subfolder images with .jpg files
and a subfolder annotations with .xml files

@param data_dir: data directory
@param objects: array containing names of objects to detect
@param model_loc: path to pretrained model
"""

data_dir="..\\tennis_data"
objects=["tennis_court"]
model_loc="..\\tennis_data\models\detection_model-ex-001--loss-0041.692.h5"






from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=data_dir)
trainer.setTrainConfig(object_names_array=objects, batch_size=16,
                       num_experiments=2, train_from_pretrained_model=model_loc)
                      # training_image_size=2048) not recoognized
trainer.trainModel()



#models sliced: 46.8677 and 44.632