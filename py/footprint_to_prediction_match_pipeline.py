import os
import shutil
from pathlib import Path
from datetime import datetime

# Import your modules
import building_footprint.Building_area_capture as step1
import yolo_model.predict as step2
import yolo_model.model_output_to_footprints as step3
"""
This code will take n number of footprints from the geojson, find the contexily imagery assosiated with it, 
crop it, run the YOLO prediction on it, and then create a csv file with the prediction, confidence level, and building area and paramiter.
If availible, the manual annoation can also be provided and added to the csv (if the annoations are done in x,y,h,w format)
"""


def run_pipeline(config):
    # --- STEP 1: IMAGE CAPTURE ---
    print("\n==========================================")
    print(f"STEP 1: Sampling {config['sample_size']} Images")
    print("==========================================")
    
    # Run the sample function
    saved_files, output_folder_path = step1.sample_and_save(
        geojson_path=config['geojson_path'],
        out_dir=config['output_base_dir'],
        n=config['sample_size'],          
        buffer_ft=config['buffer_ft'],
        seed=config['seed'],
        zoom=config['zoom'],
        crop=config['crop_images']
    )
    print(output_folder_path)

    output_folder_path = Path(output_folder_path)
    print(f"✅ Images saved to: {output_folder_path}")

    # --- STEP 2: YOLO INFERENCE ---
    print("\n==========================================")
    print("STEP 2: Running YOLO Inference")
    print("==========================================")
    
    geo_name = Path(config['geojson_path']).stem
    project_name = f"{geo_name}_seed{config['seed']}_n{config['sample_size']}_{datetime.now().strftime('%H%M%S')}"
    
    predictions_csv, labels_dir = step2.run_yolo_prediction(
        model_path=config['yolo_weights'],
        source_path=output_folder_path,
        project_name=project_name,
        save_image = config['save_image'],
        conf_threshold=config['conf_threshold']
    )
    
    print(f"✅ YOLO Predictions saved to: {predictions_csv}")

    # --- STEP 3: DATA MERGING ---
    print("\n==========================================")
    print("STEP 3: Matching and Merging Data")
    print("==========================================")
    
    # Pass the annotation directory here
    step3.Match_yolo_output_to_footprints(
        image_folder=str(output_folder_path),
        yolo_dir=labels_dir,
        predictions_csv_path=predictions_csv,
        geojson_path=config['geojson_path'],
        geojson_id_col=config['geojson_id_col'],
        annotation_dir=config.get('annotation_dir') # <--- NEW PARAMETER
    )
    
    # --- STEP 4: CLEANUP ---
    print("\n==========================================")
    print("STEP 4: Cleaning Up Intermediate Files")
    print("==========================================")

    folders_to_remove = ["building_bb", "image_size_bb", "normalized_bb"]
    
    for folder_name in folders_to_remove:
        folder_path = output_folder_path / folder_name
        if folder_path.exists():
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_name}")
            except Exception as e:
                print(f"⚠️  Could not delete {folder_name}: {e}")
        else:
            alt_path = output_folder_path / "yolo_pred" / folder_name
            if alt_path.exists():
                 shutil.rmtree(alt_path)
                 print(f"Deleted folder: {folder_name}")

    png_files = list(output_folder_path.glob("*.png"))
    for png in png_files:
        try:
            png.unlink()
        except Exception as e:
            print(f"⚠️  Could not delete image {png.name}: {e}")
            
    if png_files:
        print(f"Deleted {len(png_files)} intermediate images.")

    print("\n Pipeline complete")
    print(f"Final Data is located in: {output_folder_path / 'yolo_pred'}")

if __name__ == "__main__":
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent
    
    config = {
        # Inputs
        "geojson_path": PROJECT_ROOT / "buildingfootprint" / "provo_orem_area"/ "provo_sharon_riverbottoms.geojson", #update this path for different cities
        "yolo_weights": PROJECT_ROOT / "train16" / "weights" / "best.pt",
        "output_base_dir": PROJECT_ROOT / "output_data", 
        
        # If this is None, the annotation step is skipped.
        # If this is a path to a folder of text files (YOLO format), it will match them.
        "annotation_dir": "/Users/willicon/Desktop/provo_sharon_riverbottoms_seed15/labels"    ,  #None, # e.g., PROJECT_ROOT / "valid" / "labels",
        #Seeds must match to work correclty. 
        
        # Parameters
        "sample_size": 100, # Number of images taken of building footprints
        "buffer_ft": 50,
        "seed": 15, # Determines random seed to produce images
        "zoom": 19,
        "crop_images": True,
        "save_image": False,  #Saves predictions images. Only needed for visual verification. 
        "conf_threshold": 0.1, # The threshold for confidence of predictions
        "geojson_id_col": "id" # Don't touch
    }

    # Ensure output directory exists
    os.makedirs(config['output_base_dir'], exist_ok=True)
    
    run_pipeline(config)