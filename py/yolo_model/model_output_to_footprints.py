from pathlib import Path
import csv
import pandas as pd
import os
import geopandas as gpd 

# --------------------------
# HELPER FUNCTIONS
# --------------------------

def bbox_iou(boxA, boxB):
    """Compute IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def yolo_to_bbox(yolo_line):
    """Parses YOLO line. Returns corners for IoU and raw center/wh for CSV."""
    parts = yolo_line.strip().split()
    
    # Robust check for empty lines or malformed data
    if len(parts) != 5:
        # Check if it is just a newline or empty
        if len(parts) == 0:
            return None, None, None
        raise ValueError(f"Invalid YOLO line format (expected 5 values): {yolo_line}")
    
    class_id, x_c, y_c, w, h = map(float, parts)
    
    x_min = x_c - w/2
    y_min = y_c - h/2
    x_max = x_c + w/2
    y_max = y_c + h/2
    
    return int(class_id), (x_min, y_min, x_max, y_max), (x_c, y_c, w, h)

def normalize_latlon_bboxes(image_size_bb_dir, building_bb_dir, output_dir=None):
    image_size_bb_dir = Path(image_size_bb_dir)
    building_bb_dir = Path(building_bb_dir)
    output_dir = Path(output_dir) if output_dir else building_bb_dir.parent / "normalized_bb"
    output_dir.mkdir(parents=True, exist_ok=True)

    count_written = 0

    for req_file in building_bb_dir.glob("*.txt"):
        name = req_file.name
        image_size_file = image_size_bb_dir / name
        if not image_size_file.exists(): continue

        try:
            with open(req_file) as f:
                west_r, south_r, east_r, north_r = map(float, f.read().strip().split(","))
            with open(image_size_file) as f:
                west_t, south_t, east_t, north_t = map(float, f.read().strip().split(","))

            if east_t == west_t or north_t == south_t: continue

            norm_w = (west_r - west_t) / (east_t - west_t)
            norm_e = (east_r - west_t) / (east_t - west_t)
            norm_s = (south_r - south_t) / (north_t - south_t)
            norm_n = (north_r - south_t) / (north_t - south_t)

            out_file = output_dir / name
            with open(out_file, "w") as f:
                f.write(f"{norm_w:.6f},{norm_s:.6f},{norm_e:.6f},{norm_n:.6f}")
            count_written += 1
        except Exception as e:
            print(f"Error processing {name}: {e}")

    print(f"Normalized {count_written} files.")
    return output_dir

def match_normalized_to_yolo(normalized_bb_dir, yolo_dir, output_csv=None):
    """
    Matches normalized footprints to YOLO files.
    Includes smart filename matching to handle "UUID-ID.txt" formats.
    """
    normalized_bb_dir = Path(normalized_bb_dir)
    yolo_dir = Path(yolo_dir)
    
    if output_csv is None:
        parent_dir = normalized_bb_dir.parent
        out_folder = parent_dir / "matched_output"
        out_folder.mkdir(parents=True, exist_ok=True)
        output_csv = out_folder / "matched_classes_temp.csv"
    else:
        output_csv = Path(output_csv)
        out_folder = output_csv.parent
        out_folder.mkdir(parents=True, exist_ok=True)

    results = []

    # --- 1. BUILD FILE MAPPING ---
    # We scan the YOLO directory once and create a map: { "ID": "full_path" }
    # This allows us to find "0bf95cdd-1722.txt" when we look up "1722"
    yolo_file_map = {}
    
    if yolo_dir.exists():
        for y_file in yolo_dir.glob("*.txt"):
            # Store exact match (e.g., "1722")
            yolo_file_map[y_file.stem] = y_file
            
            # Store ID match (strip prefix before last hyphen)
            # e.g., "0bf95cdd-1722" -> "1722"
            if '-' in y_file.stem:
                clean_id = y_file.stem.split('-')[-1]
                yolo_file_map[clean_id] = y_file
    else:
        print(f"⚠️ Warning: YOLO/Annotation directory not found: {yolo_dir}")

    # --- 2. PERFORM MATCHING ---
    for norm_file in normalized_bb_dir.glob("*.txt"):
        image_name = norm_file.stem # This is the building ID, e.g., "1722"
        
        # Look up the file in our smart map
        yolo_file = yolo_file_map.get(image_name)

        if not yolo_file: 
            # Silent skip or debug print if needed
            continue

        norm_vals = list(map(float, norm_file.read_text().strip().split(",")))
        norm_bbox = (norm_vals[0], norm_vals[1], norm_vals[2], norm_vals[3])

        best_iou = -1
        best_class = None
        best_xywh = (0,0,0,0)

        try:
            with open(yolo_file) as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        class_id, yolo_bbox_corners, yolo_raw_xywh = yolo_to_bbox(line)
                        if class_id is None: continue # Skip empty lines

                        iou = bbox_iou(norm_bbox, yolo_bbox_corners)
                        if iou > best_iou:
                            best_iou = iou
                            best_class = class_id
                            best_xywh = yolo_raw_xywh
                    except ValueError:
                        continue 
        except Exception as e:
            print(f"Error reading {yolo_file.name}: {e}")
            continue

        if best_class is not None and best_iou > 0:
            results.append((image_name, best_class, best_xywh[0], best_xywh[1], best_xywh[2], best_xywh[3]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "class_id", "norm_x_center", "norm_y_center", "norm_width", "norm_height"])
        writer.writerows(results)

    print(f"Matching complete. Saved {len(results)} matches to {output_csv.name}")
    return output_csv

def process_geojson_metrics(geojson_path, id_column_name="ID"):
    print(f"Loading GeoJSON from: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    
    # Estimate UTM CRS and project
    gdf_projected = gdf.to_crs(gdf.estimate_utm_crs())
    
    gdf_projected['Area_sqm'] = gdf_projected.geometry.area
    gdf_projected['Perimeter_m'] = gdf_projected.geometry.length
    
    if id_column_name not in gdf_projected.columns:
        raise ValueError(f"Column '{id_column_name}' not found in GeoJSON.")
    
    metrics_df = gdf_projected[[id_column_name, 'Area_sqm', 'Perimeter_m']].copy()
    
    metrics_df[id_column_name] = metrics_df[id_column_name].astype(str).apply(lambda x: x.replace('.0', '') if x.endswith('.0') else x)
    
    if id_column_name != 'ID':
        metrics_df.rename(columns={id_column_name: 'ID'}, inplace=True)
        
    print(f"Calculated metrics for {len(metrics_df)} buildings.")
    return metrics_df

def add_confidence_clean_and_merge_metrics(matched_csv_path, predictions_csv_path, geojson_path, geojson_id_col, output_csv_path, annotation_match_csv=None):
    df_matched = pd.read_csv(matched_csv_path)
    df_preds = pd.read_csv(predictions_csv_path)
    
    # Clean Filenames
    df_preds['Clean_Filename'] = df_preds['Source_File'].astype(str).apply(lambda x: os.path.splitext(x)[0])
    
    confidences = []
    MAX_ALLOWED_DIFF = 0.02 

    # Match Confidence
    for index, row in df_matched.iterrows():
        img_name = str(row['image_name'])
        if img_name.endswith('.0'):
            img_name = img_name[:-2] 
        
        subset = df_preds[df_preds['Clean_Filename'] == img_name]
        
        if subset.empty:
            confidences.append(None)
            continue
            
        best_conf = None
        min_diff = float('inf')

        for _, pred_row in subset.iterrows():
            diff = (abs(row['norm_x_center'] - pred_row['Normalized_X_Center']) +
                    abs(row['norm_y_center'] - pred_row['Normalized_Y_Center']) +
                    abs(row['norm_width'] - pred_row['Normalized_Width']) +
                    abs(row['norm_height'] - pred_row['Normalized_Height']))
            
            if diff < min_diff:
                min_diff = diff
                if diff < MAX_ALLOWED_DIFF:
                    best_conf = pred_row['Confidence']

        confidences.append(best_conf)

    df_matched['Confidence'] = confidences
    
    # Merge Annotations (OPTIONAL)
    if annotation_match_csv and os.path.exists(annotation_match_csv):
        print(f"Merging annotations from {annotation_match_csv}...")
        try:
            df_ann = pd.read_csv(annotation_match_csv)
            if not df_ann.empty:
                # Keep only relevant columns and rename
                df_ann = df_ann[['image_name', 'class_id']]
                df_ann.rename(columns={'class_id': 'annotation_type'}, inplace=True)
                
                # Merge onto the matched dataframe
                df_matched = pd.merge(df_matched, df_ann, on='image_name', how='left')
            else:
                print("⚠️ Annotation CSV was empty.")
        except Exception as e:
            print(f"⚠️ Failed to merge annotations: {e}")
    
    # Cleanup & Rename
    cols_to_drop = ['norm_x_center', 'norm_y_center', 'norm_width', 'norm_height']
    df_final = df_matched.drop(columns=cols_to_drop)
    
    df_final.rename(columns={'image_name': 'ID'}, inplace=True)
    df_final.rename(columns={'class_id': 'yolo_pred'}, inplace=True)
    
    df_final['ID'] = df_final['ID'].astype(str).apply(lambda x: x.replace('.0', '') if x.endswith('.0') else x)

    # Merge GeoJSON Metrics
    if geojson_path and os.path.exists(geojson_path):
        metrics_df = process_geojson_metrics(geojson_path, geojson_id_col)
        df_final = pd.merge(df_final, metrics_df, on='ID', how='left')
        print("Merged Area and Perimeter data.")
    else:
        print("⚠️ GeoJSON path not found. Skipping metrics.")

    df_final.to_csv(output_csv_path, index=False)
    matches_found = df_final['Confidence'].notna().sum()
    print(f"Final CSV saved to: {output_csv_path}")
    print(f"Statistics: Found confidence for {matches_found} items.")

# --------------------------
# MAIN EXECUTION FLOW
# --------------------------

def Match_yolo_output_to_footprints(image_folder, yolo_dir, predictions_csv_path, geojson_path, geojson_id_col="ID", annotation_dir=None):
    print("--- Step 1: Normalizing Lat/Lon Bounding Boxes ---")
    image_size_bb_folder = f"{image_folder}/image_size_bb"
    building_bb_folder = f"{image_folder}/building_bb"
    
    normalized_bb_dir = normalize_latlon_bboxes(image_size_bb_folder, building_bb_folder)
    
    print("\n--- Step 2: Matching Footprints to YOLO Predictions ---")
    temp_pred_csv_path = Path(image_folder) / "yolo_pred" / "temp_yolo_preds.csv"
    final_output_path = Path(image_folder) / "yolo_pred" / "yolo_output_data.csv"

    match_normalized_to_yolo(normalized_bb_dir, yolo_dir, output_csv=temp_pred_csv_path)

    # Optional: Match Annotations
    temp_ann_csv_path = None
    if annotation_dir:
        print("\n--- Step 2b: Matching Footprints to Annotations ---")
        if os.path.exists(annotation_dir):
            temp_ann_csv_path = Path(image_folder) / "yolo_pred" / "temp_yolo_annotations.csv"
            match_normalized_to_yolo(normalized_bb_dir, annotation_dir, output_csv=temp_ann_csv_path)
        else:
            print(f"⚠️ Annotation directory provided but not found: {annotation_dir}")

    print("\n--- Step 3: Merging Confidence, Annotations, Area, and Perimeter ---")
    add_confidence_clean_and_merge_metrics(
        matched_csv_path=temp_pred_csv_path, 
        predictions_csv_path=predictions_csv_path, 
        geojson_path=geojson_path, 
        geojson_id_col=geojson_id_col, 
        output_csv_path=final_output_path,
        annotation_match_csv=temp_ann_csv_path 
    )