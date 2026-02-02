from ultralytics import YOLO

name = "orem_windsor"
model = YOLO("/Users/willicon/Desktop/dronemodeling_Logan/train18_provo_orem/weights/best.pt") 

results = model.val(data= "py/yolo_model/config_val.yaml" #Uses validation congfig.
                    ,name = f"{name}_validation"
                    ,device="mps") #optimized to run on a mac processor