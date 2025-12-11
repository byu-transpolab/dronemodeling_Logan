from ultralytics import YOLO
import pandas as pd
import os

model = YOLO("/Users/willicon/Desktop/dronemodeling_Logan/train16/weights/best.pt") # this is the path to the model you want to run from your training
name = "stgeorge_run"
save_output = True # This will save the prediction output in a excel file on the desktop with the following: image-label(s)-confidence level

results = model.predict(source= "/Users/willicon/Desktop/Buildings_seed25_2025_12_01_150531_stgeorge_kmeans_cluster.geojson", # This can be a single image or a file
            name = name, # Name the output folder
            show = False,          # Will open up photo as it predicts
            save = True,           # Will save picture results in predict file
            conf = 0.6,            # This will only display detections to this confidence 
            line_width = 1,        # Changes how big the annotation text is
            save_crop = False,     # Will crop to each detection box
            save_txt = True,       # Save bounded boxed as txt file (reverse annotations)
            show_labels = True,    # Show the labels for the bounding box
            show_conf = True,      # Shows the confidence labels for bounding box
            classes = [0,1,2,3,4,5]    # Runs detection for each class in list
            )

            #https://docs.ultralytics.com/modes/predict/#inference-arguments See for other arguments


if save_output:
    # 2. Extract Data and Prepare for Excel
    all_detection_data = []

    # Get the class names from the model's metadata (e.g., {'0': 'building', '1': 'car', ...})
    class_names = model.names 

    # Iterate through each image/source result
    for result in results:
        # Get the file name of the source image/data
        if result.path:
            source_file = os.path.basename(result.path) 
        else:
            source_file = "N/A" # Fallback if path is somehow missing
        
        # Extract the bounding box coordinates (xyxy format: top-left x, top-left y, bottom-right x, bottom-right y)
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Iterate through all detected objects in the current image
        for i in range(len(boxes)):
            # Extract coordinates and convert to standard floats/integers
            x1, y1, x2, y2 = [round(float(coord), 2) for coord in boxes[i]]
            confidence = round(float(confidences[i]), 4)
            class_id = class_ids[i]
            class_name = class_names.get(class_id, f"Class_{class_id}") # Get the name from the ID
            
            # Append all data for this single detection to the main list
            all_detection_data.append({
                'Source_File': source_file,
                'Class_ID': class_id,
                'Class_Name': class_name,
                'Confidence': confidence,
                'Box_X_Min': x1,
                'Box_Y_Min': y1,
                'Box_X_Max': x2,
                'Box_Y_Max': y2
            })

    # 3. Create DataFrame and Save to Excel
    if all_detection_data:
        df = pd.DataFrame(all_detection_data)
        
        # Define the output path for your Excel file
        excel_output_path = f"/Users/willicon/Desktop/{name}.xlsx"
        
        # Save the DataFrame to an Excel file
        df.to_excel(excel_output_path, index=False)
        
        print(f"\n✅ Successfully saved {len(df)} detections to: {excel_output_path}")
    else:
        print("\n⚠️ No detections found with the given confidence threshold (conf=0.6).")