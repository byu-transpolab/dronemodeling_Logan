"""
sample_buildings_and_save_imagery.py

Reads a GeoJSON of building footprints (Polygons), randomly samples up to N features,
buffers each by 100 ft, fetches aerial imagery around the buffer using contextily/Esri
World Imagery, and saves each image to disk.

Usage:
    python sample_buildings_and_save_imagery.py \
        --geojson buildings.geojson \
        --out-dir ./out_images \
        --n 100 \
        --seed 42
"""

import os
import argparse
import random
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import contextily as ctx
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import datetime

# Convert feet to meters
FEET_TO_METERS = 0.3048

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def sample_and_save(geojson_path, out_dir, n=100, buffer_ft=100, seed=None, zoom=19):
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_dir_ts = Path(out_dir) / f"building_seed{seed}_{timestamp}"
    ensure_dir(out_dir_ts)
    
    print(f"Saving images to: {out_dir_ts}")
    
    # Use out_dir_ts instead of out_dir for saving files
    out_dir = out_dir_ts
    
    # Read geojson
    gdf = gpd.read_file(geojson_path)

    # Keep only polygon-like geometries
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No polygon geometries found in the GeoJSON.")

    # Sample up to n features
    if seed is not None:
        random.seed(seed)
    count = min(n, len(gdf))
    sampled_idx = random.sample(list(gdf.index), k=count)
    sampled = gdf.loc[sampled_idx].reset_index(drop=True)

    # Ensure projection: buffer in meters using Web Mercator (EPSG:3857)
    sampled_3857 = sampled.to_crs(epsg=3857)

    buffer_m = buffer_ft * FEET_TO_METERS

    ensure_dir(out_dir)
    saved = []

    for i, row in sampled_3857.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Buffer by 100 ft (converted to meters)
            buffered = geom.buffer(buffer_m)

            # Convert buffered polygon back to WGS84 (lon/lat) for tile bounding box
            buffered_wgs = gpd.GeoSeries([buffered], crs=sampled_3857.crs).to_crs(epsg=4326).iloc[0]

            # Get bounding box (west, south, east, north) in lon/lat
            minx, miny, maxx, maxy = buffered_wgs.bounds

            # Slightly pad bbox to avoid tight cropping (optional)
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            west, south, east, north = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

            # Fetch tiles as an image using contextily
            # ctx.bounds2img expects (west, south, east, north) with ll=True for lon/lat
            try:
                img_arr, ext = ctx.bounds2img(west, south, east, north, zoom=zoom, ll=True,
                                              source=ctx.providers.Esri.WorldImagery)
            except Exception as e:
                # On failure (e.g., zoom too high), try lower zooms
                print(f"[{i}] bounds2img failed at zoom {zoom}: {e}. Trying lower zooms.")
                success = False
                for z in range(zoom-1, max(10, zoom-6), -1):
                    try:
                        img_arr, ext = ctx.bounds2img(west, south, east, north, zoom=z, ll=True,
                                                      source=ctx.providers.Esri.WorldImagery)
                        success = True
                        print(f"[{i}] succeeded at zoom {z}")
                        break
                    except Exception:
                        continue
                if not success:
                    print(f"[{i}] failed to fetch tiles for bbox {west,south,east,north}. Skipping.")
                    continue

            # img_arr is (3, H, W) in uint8 (RGB)
            # Convert to image (H, W, 3)
            if img_arr.ndim == 3 and img_arr.shape[0] in (3,4):
                # transpose from (channels, H, W) to (H, W, C)
                img = np.transpose(img_arr, (1, 2, 0))
            else:
                img = img_arr

            # Save with PIL for robust format support
            pil_img = Image.fromarray(img)

            # Create a meaningful filename: idx + optional id from properties
            prop_id = None
            if isinstance(row.get('id'), (str,int)):
                prop_id = row.get('id')
            else:
                # try common properties
                for candidate in ['id', 'fid', 'uid', 'building_id', 'osm_id']:
                    if candidate in row.index:
                        prop_id = row[candidate]
                        break

            name_fragment = f"{i}" if prop_id is None else f"{i}_{prop_id}"
            filename = Path(out_dir) / f"building_{name_fragment}.png"

            pil_img.save(filename)
            saved.append(str(filename))
            print(f"[{i}] saved {filename}")
        except Exception as exc:
            print(f"[{i}] unexpected error: {exc}. Skipping.")

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", required=True, help="Path to input GeoJSON file of building footprints")
    parser.add_argument("--out-dir", default="./out_images", help="Directory to save images")
    parser.add_argument("--n", type=int, default=100, help="Number of footprints to randomly sample")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--buffer-ft", type=float, default=100.0, help="Buffer distance in feet around polygon")
    parser.add_argument("--zoom", type=int, default=19, help="Initial zoom level for imagery (higher = more detail)")
    args = parser.parse_args()

    saved = sample_and_save(args.geojson, args.out_dir, n=args.n, buffer_ft=args.buffer_ft, seed=args.seed, zoom=args.zoom)
    print(f"Done. Saved {len(saved)} images to {args.out_dir}")

geojson_path = "/Users/willicon/Desktop/dronemodeling_Logan/buildingfootprint/logan.geojson"
out_dir = "/Users/willicon/Desktop"

#Zoom 19 is closest zoom we can get
#The seed tells what bulding to sampele. Remebering the seed will allow it to be reproduced. 
sample_and_save(geojson_path, out_dir, n=100, buffer_ft=50, zoom=19, seed=30)