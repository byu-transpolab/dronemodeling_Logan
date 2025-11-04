from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt") # this is the path to the model you want to run from your training

model.predict(source="/Users/willicon/Desktop/seed80_2025_11_04_133501", # This can be a single image or a file
            show = False,       # Will open up photo
            save = True,       # Will save results in predict file
            conf = 0.6,        # This will only display detections to this confidence 
            line_width = 1,    # Changes how big the annotation text is
            save_crop = False,  # Will crop to each detection box
            save_txt = True,   # Save bounded boxed as txt file (reverse annotations)
            show_labels = True, # Show the labels for the bounding box
            show_conf = True,  # Shows the confidence labels for bounding box
            classes = [0,1,2,3,4,5]    # Runs detection for each class in list
            )