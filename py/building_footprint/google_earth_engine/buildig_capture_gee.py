import ee
import geemap
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image

# 1. Initialize Earth Engine
# Ensure your Cloud Project ID is correct
PROJECT_ID = 'provo-test' 

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

def run_provo_capture(center_coords, out_dir, n=50, buffer_ft=50):
    # Setup Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"Provo_HighRes_n{n}_{timestamp}"
    save_path = Path(out_dir) / folder_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    log_data = [] # List to store our log entries

    # 2. Define Data Sources
    # Buffer the center to ensure 5km of coverage for sampling
    aoi = ee.Geometry.Point(center_coords).buffer(5000).bounds()
    
    # Combined Microsoft/Google Footprints
    buildings = ee.FeatureCollection("projects/sat-io/open-datasets/VIDA_COMBINED/USA").filterBounds(aoi)
    
    # NAIP Imagery: Mosaic for gap-filling and uint8 for PIL compatibility
    naip = (ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(aoi)
            .filterDate('2021-01-01', '2025-12-31')
            .mosaic()
            .uint8() 
            .select(['R', 'G', 'B']))

    # 3. Sample Buildings
    sampled_fc = buildings.randomColumn('random').sort('random').limit(n)
    features = sampled_fc.getInfo()['features']

    print(f"Starting High-Res Download of {len(features)} buildings...")
    print(f"Saving to: {save_path}")

    for i, feat in enumerate(features):
        # Coordinates for the log
        geom = ee.Geometry(feat['geometry'])
        coords = geom.centroid().getInfo()['coordinates'] # [lon, lat]
        
        # Buffer ROI
        roi = geom.bounds().buffer(buffer_ft * 0.3048).bounds()
        
        fid = feat['id'].split('/')[-1] if 'id' in feat else i
        tif_file = save_path / f"{fid}.tif"
        png_file = save_path / f"{fid}.png"

        # Throttle to stay under GEE rate limits
        time.sleep(2) 

        success = False
        attempts = 0
        status_msg = "Failed"

        while not success and attempts < 3:
            try:
                # Force scale=0.6 for native resolution
                geemap.ee_export_image(naip, filename=str(tif_file), scale=0.6, region=roi, file_per_band=False)
                
                if tif_file.exists():
                    with Image.open(tif_file) as img:
                        # Save with zero compression to maintain clarity
                        img.save(png_file, format='PNG', compress_level=0)
                    tif_file.unlink() # Cleanup
                    print(f"[{i+1}/{n}] Success: {fid}")
                    success = True
                    status_msg = "Success"
            except Exception as e:
                attempts += 1
                print(f"[{i+1}/{n}] Attempt {attempts} failed for {fid}: {e}")
                time.sleep(4)

        # Update log data
        log_data.append({
            'building_id': fid,
            'longitude': coords[0],
            'latitude': coords[1],
            'status': status_msg,
            'image_path': str(png_file) if success else "N/A"
        })

    # 4. Save the Log File
    log_df = pd.DataFrame(log_data)
    log_csv = save_path / "capture_log.csv"
    log_df.to_csv(log_csv, index=False)
    
    return save_path, log_csv

#------------#
# Execution
#------------#
if __name__ == "__main__":
    # Centered on Provo, UT
    PROVO_COORDS = [-111.6585, 40.2338] 
    DESKTOP = os.path.expanduser("~/Desktop")

    final_folder, final_log = run_provo_capture(PROVO_COORDS, DESKTOP, n=20)
    
    print("\n" + "="*30)
    print(f"PROCESS COMPLETE")
    print(f"Images in: {final_folder}")
    print(f"Log File: {final_log}")
    print("="*30)