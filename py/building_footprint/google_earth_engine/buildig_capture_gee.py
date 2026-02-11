import ee
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import requests

# 1. Initialize
PROJECT_ID = 'provo-test' 
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

def run_provo_final(center_coords, out_dir, n=50, buffer_ft=50):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(out_dir) / f"Provo_Native_Res_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Data
    aoi = ee.Geometry.Point(center_coords).buffer(5000).bounds()
    buildings = ee.FeatureCollection("projects/sat-io/open-datasets/VIDA_COMBINED/USA").filterBounds(aoi)
    
    # Load NAIP and select ONLY the RGB bands immediately
    naip_collection = (ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(aoi)
            .filterDate('2021-01-01', '2025-12-31')
            .mosaic()) # Mosaic to fill gaps

    # 3. Sample
    sampled_fc = buildings.randomColumn('random').sort('random').limit(n)
    features = sampled_fc.getInfo()['features']
    log_data = []

    print(f"Starting Native-Scale Download for {n} buildings...")
    print(f"Saving to: {save_path}")

    for i, feat in enumerate(features):
        geom = ee.Geometry(feat['geometry'])
        centroid = geom.centroid().getInfo()['coordinates']
        
        # Define the exact crop region
        roi = geom.bounds().buffer(buffer_ft * 0.3048).bounds()
        fid = feat['id'].split('/')[-1] if 'id' in feat else i
        png_file = save_path / f"building_{fid}.png"

        time.sleep(1) # Slight throttle

        try:
            # THE FIX: Use .visualize()
            # This forces the image into 3 bands (RGB) and scales values to 0-255 (uint8)
            # strictly for display/export purposes.
            visualized_image = naip_collection.visualize(
                bands=['R', 'G', 'B'],
                min=0,
                max=255
            )
            
            # Request the download URL for the visualized image
            url = visualized_image.getDownloadURL({
                'scale': 0.6,          # Native 60cm resolution
                'crs': 'EPSG:3857',    # Web Mercator (Square pixels)
                'region': roi,
                'format': 'png'
            })
            
            # Download and write
            response = requests.get(url)
            if response.status_code == 200:
                with open(png_file, 'wb') as f:
                    f.write(response.content)
                print(f"[{i+1}/{n}] Saved: {fid}")
                status = "Success"
            else:
                print(f"[{i+1}/{n}] Server Error {fid}: {response.text}")
                status = "Server Error"

        except Exception as e:
            print(f"[{i+1}/{n}] Failed {fid}: {e}")
            status = "Fail"

        log_data.append({'id': fid, 'lon': centroid[0], 'lat': centroid[1], 'status': status})

    # Save Log
    pd.DataFrame(log_data).to_csv(save_path / "capture_log.csv", index=False)
    return save_path

if __name__ == "__main__":
    PROVO_COORDS = [-111.6585, 40.2338] 
    DESKTOP = os.path.expanduser("~/Desktop")
    run_provo_final(PROVO_COORDS, DESKTOP, n=20)