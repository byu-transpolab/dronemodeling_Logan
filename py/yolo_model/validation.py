from ultralytics import YOLO

name = "saratoga_springs_2"
model = YOLO("/Users/willicon/Desktop/dronemodeling_Logan/train16/weights/last.pt") 

results = model.val(data= "/Users/willicon/Desktop/dronemodeling_Logan/py/ultralytics_model/config_val.yaml" #Uses validation congfig.
                    ,name = f"{name}_validation"
                    ,device="mps") #optimized to run on a mac processor