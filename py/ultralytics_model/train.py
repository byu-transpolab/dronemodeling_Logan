from ultralytics import YOLO

#load a model
model= YOLO("yolov8n.yaml")

#use the model
results = model.train(data="py/ultralytics_model/config.yaml", epochs =100) #train the model