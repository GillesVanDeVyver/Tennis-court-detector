"""
This script compares all models in a given directory by calculating
the mean Average Precicion(mAP) of every model on the given data with given
Intersection of Union (IoU) threshold,
Non-Maximum Suppression (nms) threshold and
object threshold

@param data_dir: data directory
@param model_dir: model directory
@param config_dir: json config directory

@param IoU_thresh: IoU threshold
@param obj_thresh: object threshold
@param nms_thresh: nms threshold
"""

data_dir="../tennis_data"
model_dir="../tennis_data/models"
config_dir="../tennis_data/json/detection_config.json"

IoU_thresh=0.3
obj_thresh=0.3
nms_thresh=0.3








from imageai.Detection.Custom import DetectionModelTrainer


trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=data_dir)
trainer.evaluateModel(model_path=model_dir, json_path=config_dir, iou_threshold=IoU_thresh,
                      object_threshold=nms_thresh, nms_threshold=nms_thresh)