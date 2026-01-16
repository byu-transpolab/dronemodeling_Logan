import pandas as pd
import numpy as np
import os

def add_confidence_and_clean_debug(matched_csv_path, predictions_csv_path, output_csv_path):
    """
    Debug version: Prints exactly why matches are failing.
    """
    print(f"--- LOADING DATA ---")
    print(f"Matched CSV: {matched_csv_path}")
    print(f"Predictions CSV: {predictions_csv_path}")
    
    df_matched = pd.read_csv(matched_csv_path)
    df_preds = pd.read_csv(predictions_csv_path)

    print(f"Loaded {len(df_matched)} matched rows.")
    print(f"Loaded {len(df_preds)} prediction rows.")
    
    # 1. CHECK COLUMN NAMES
    print("\n--- CHECKING COLUMNS ---")
    print(f"Matched Columns: {list(df_matched.columns)}")
    print(f"Prediction Columns: {list(df_preds.columns)}")
    
    expected_col = 'Normalized_X_Center'
    if expected_col not in df_preds.columns:
        print(f"❌ CRITICAL ERROR: Prediction CSV is missing '{expected_col}'.")
        print("   It looks like you might be using the OLD pixel-based CSV.")
        print("   Please re-run the YOLO prediction code (Step 1) to generate the normalized CSV.")
        return

    confidences = []
    MAX_ALLOWED_DIFF = 0.02 # Very loose tolerance (2%)

    print("\n--- DIAGNOSING FIRST 5 ROWS ---")
    
    for index, row in df_matched.iterrows():
        img_name = str(row['image_name'])
        
        # DEBUG: Print the first few attempts
        if index < 5:
            print(f"\nProcessing Row {index}: Image '{img_name}'")
            print(f"   Target Coords: x={row['norm_x_center']:.4f}, y={row['norm_y_center']:.4f}")

        # 1. Filter predictions
        subset = df_preds[df_preds['Source_File'].astype(str).str.contains(img_name, regex=False)]
        
        if subset.empty:
            if index < 5:
                print(f"   ❌ NO MATCHING FILENAME FOUND in predictions.")
                print(f"      (Checked against first 3 CSV files: {df_preds['Source_File'].head(3).tolist()})")
            confidences.append(None)
            continue
            
        # 2. Find Closest Box
        best_conf = None
        min_diff = float('inf')
        closest_match_details = ""

        for _, pred_row in subset.iterrows():
            # Calculate difference
            d_x = abs(row['norm_x_center'] - pred_row['Normalized_X_Center'])
            d_y = abs(row['norm_y_center'] - pred_row['Normalized_Y_Center'])
            d_w = abs(row['norm_width'] - pred_row['Normalized_Width'])
            d_h = abs(row['norm_height'] - pred_row['Normalized_Height'])
            
            total_diff = d_x + d_y + d_w + d_h
            
            if total_diff < min_diff:
                min_diff = total_diff
                if total_diff < MAX_ALLOWED_DIFF:
                    best_conf = pred_row['Confidence']
                
                # Capture details for debug print
                closest_match_details = (f"Pred Coords: x={pred_row['Normalized_X_Center']:.4f}, "
                                         f"y={pred_row['Normalized_Y_Center']:.4f} | Diff: {total_diff:.4f}")

        if best_conf is not None:
            if index < 5:
                print(f"   ✅ MATCH FOUND! Confidence: {best_conf}")
            confidences.append(best_conf)
        else:
            if index < 5:
                print(f"   ❌ COORDINATE MISMATCH.")
                print(f"      Closest candidate was: {closest_match_details}")
                print(f"      Allowed Diff: {MAX_ALLOWED_DIFF}")
                if min_diff > 1.0:
                    print("      ⚠️ HUGE DIFFERENCE DETECTED. Are you comparing Pixels vs Normalized?")
            confidences.append(None)

    # Save
    df_matched['Confidence'] = confidences
    cols_to_drop = ['norm_x_center', 'norm_y_center', 'norm_width', 'norm_height']
    df_final = df_matched.drop(columns=cols_to_drop)
    
    df_final.to_csv(output_csv_path, index=False)
    
    success_count = df_final['Confidence'].notna().sum()
    print(f"\n--- DONE ---")
    print(f"Saved to: {output_csv_path}")
    print(f"Total Matches Found: {success_count} / {len(df_final)}")
    if success_count == 0:
        print("⚠️ STILL ZERO MATCHES. Check the Diagnosis output above!")

# --- RUN CONFIGURATION ---
matched_csv_path = "/Users/willicon/Desktop/Buildings_seed30_2025_11_06_084516/matched_output/temp_matched_with_coords.csv"
predictions_csv_path = "/Users/willicon/Desktop/dronemodeling_Logan/runs/detect/logan_seed30/logan_seed30.csv"
final_output = "/Users/willicon/Desktop/Buildings_seed30_2025_11_06_084516/matched_output/matched_classes_final.csv"

# Run the debug function
add_confidence_and_clean_debug(matched_csv_path, predictions_csv_path, final_output)