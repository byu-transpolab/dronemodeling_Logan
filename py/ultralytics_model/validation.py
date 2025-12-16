from ultralytics import YOLO

name = "logan"
model = YOLO("/Users/willicon/Desktop/dronemodeling_Logan/train16/weights/last.pt") 

results = model.val(data= "/Users/willicon/Desktop/dronemodeling_Logan/py/ultralytics_model/config_val.yaml"
                    ,name = f"{name}_validation"
                    ,device="mps")