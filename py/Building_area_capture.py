import argparse
import random
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import contextily as ctx
from PIL import Image
import numpy as np

# Convert feet to meters
FEET_TO_METERS = 0.3048


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def sample_and_save(geojson_path, out_dir, n, buffer_ft, seed, zoom, clusters=None):
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    subfolder_name = f"seed{seed}_{timestamp}"
    if clusters:
        cluster_str = "_".join(map(str, clusters))
        subfolder_name = f"clusters_{cluster_str}_{subfolder_name}"
    out_dir_ts = Path(out_dir) / subfolder_name
    ensure_dir(out_dir_ts)

    # Create folder for bounding box txt files
    bb_dir = out_dir_ts / "lat_log_bb"
    ensure_dir(bb_dir)

    print(f"Saving images to: {out_dir_ts}")
    print(f"Saving bounding boxes to: {bb_dir}")

    # Read GeoJSON
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.reset_index(drop=True)

    if gdf.empty:
        raise ValueError("No valid polygon geometries found in the GeoJSON.")

    # Filter by cluster types if provided
    if clusters is not None:
        if "cluster" not in gdf.columns:
            raise ValueError("GeoJSON does not contain a 'cluster' property to filter on.")
        gdf = gdf[gdf["cluster"].isin(clusters)]
        print(f"Filtered GeoJSON to {len(gdf)} features in clusters {clusters}.")

    if gdf.empty:
        raise ValueError(f"No polygons found after filtering by clusters {clusters}.")

    # Sample up to n features
    if seed is not None:
        random.seed(seed)
    count = min(n, len(gdf))
    sampled_idx = random.sample(list(gdf.index), k=count)
    sampled = gdf.loc[sampled_idx].reset_index(drop=True)

    # Ensure projection: buffer in meters using Web Mercator (EPSG:3857)
    sampled_3857 = sampled.to_crs(epsg=3857)
    buffer_m = buffer_ft * FEET_TO_METERS

    saved = []

    for i, row in sampled_3857.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Buffer polygon
            buffered = geom.buffer(buffer_m)

            # Convert buffered polygon to WGS84 (lon/lat)
            buffered_wgs = gpd.GeoSeries([buffered], crs=sampled_3857.crs).to_crs(epsg=4326).iloc[0]

            # Get bounding box
            minx, miny, maxx, maxy = buffered_wgs.bounds
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            west, south, east, north = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

            # Determine file name
            prop_id = None
            if isinstance(row.get('id'), (str, int)):
                prop_id = row.get('id')
            elif "id" in row.index:
                prop_id = row["id"]
            elif "fid" in row.index:
                prop_id = row["fid"]

            name_fragment = f"{prop_id if prop_id is not None else i}"
            filename = Path(out_dir_ts) / f"{name_fragment}.png"
            bb_filename = bb_dir / f"{name_fragment}.txt"

            # Save bounding box to txt
            with open(bb_filename, "w") as f:
                f.write(f"{west},{south},{east},{north}")

            # Fetch tiles as an image
            try:
                img_arr, ext = ctx.bounds2img(west, south, east, north, zoom=zoom, ll=True,
                                              source=ctx.providers.Esri.WorldImagery)
            except Exception as e:
                print(f"[{i}] bounds2img failed at zoom {zoom}: {e}. Trying lower zooms.")
                success = False
                for z in range(zoom - 1, max(10, zoom - 6), -1):
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

            # Convert to RGB image
            if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
                img = np.transpose(img_arr, (1, 2, 0))
            else:
                img = img_arr

            pil_img = Image.fromarray(img)
            pil_img.save(filename)

            saved.append(str(filename))
            print(f"[{i}] saved image {filename} and bounding box {bb_filename}")

        except Exception as exc:
            print(f"[{i}] unexpected error: {exc}. Skipping.")

    return saved

geojson_path = "/Users/willicon/Desktop/dronemodeling_Logan/buildingfootprint/logan_kmeans_cluster.geojson" #"/Users/willicon/Desktop/dronemodeling_Logan/buildingfootprint/logan.geojson"
out_dir = "/Users/willicon/Desktop"


#Zoom 19 is closest zoom we can get
#The seed tells what bulding to sampele. Remebering the seed will allow it to be reproduced. 
sample_and_save(geojson_path, 
                out_dir, 
                n=100, 
                buffer_ft=10, 
                zoom=19, 
                seed=80 )
                #clusters=[1,2,3]) # Clusters are given as list