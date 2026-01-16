import random
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import contextily as ctx
from PIL import Image
import numpy as np
from shapely.geometry import box

# Convert feet to meters
FEET_TO_METERS = 0.3048


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def crop_images_by_latlon(image_folder, true_latlon_folder, crop_latlon_folder):
    """
    Crop images using a crop bounding box in lat/lon and save them
    in the same folder as the input images (overwriting original images).
    
    Parameters:
    - image_folder: folder with original images
    - true_latlon_folder: folder with txt files describing the lat/lon of the full image (west,south,east,north)
    - crop_latlon_folder: folder with txt files describing the crop bounding box (west,south,east,north)
    """
    image_folder = Path(image_folder)
    image_txt_folder = Path(true_latlon_folder)
    crop_txt_folder = Path(crop_latlon_folder)

    for img_file in image_folder.glob("*.png"):
        img_name = img_file.stem
        img_txt_file = image_txt_folder / f"{img_name}.txt"
        crop_txt_file = crop_txt_folder / f"{img_name}.txt"

        if not img_txt_file.exists() or not crop_txt_file.exists():
            print(f"Skipping {img_name}, missing txt file.")
            continue

        # Load image
        img = Image.open(img_file)
        width, height = img.size

        # Read full image bbox
        with open(img_txt_file) as f:
            w, s, e, n = map(float, f.read().strip().split(","))

        # Read crop bbox
        with open(crop_txt_file) as f:
            cw, cs, ce, cn = map(float, f.read().strip().split(","))

        # Convert lat/lon to pixel coordinates
        def lon_to_px(lon):
            return int(round((lon - w) / (e - w) * width))

        def lat_to_py(lat):
            return int(round((n - lat) / (n - s) * height))

        left = lon_to_px(cw)
        upper = lat_to_py(cn)
        right = lon_to_px(ce)
        lower = lat_to_py(cs)

        # Crop and save (overwrite original image)
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img.save(img_file)
    
    print("Images croped to building footprint")

def sample_and_save(geojson_path, out_dir, n, buffer_ft, seed, zoom=19, crop = True, clusters=None):
    """
    Creates 'n' number of aerial images based off of the building footprints in a geojson file. Can be croped to only include the building. 
    Images are named based off of the building footprint ID.
    
    Parameters:
    - geojson_path: The path to the geojson file with the building footprint.
    - out_dir: The path to where the image folder will be created.
    - n: How many images to create.
    - buffer_ft: Buffer zone around building. 
    - seed: What random collection is used. Same seed give the same images
    - zoom=19: How close to the building the image will be. 19 will give the closest zoom
    - crop = True: Crop the images to only include the building with the building 
    - clusters=None
    """

    location =Path(geojson_path).stem
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    subfolder_name = f"{location}_seed{seed}_{timestamp}"
    if clusters:
        cluster_str = "_".join(map(str, clusters))
        subfolder_name = f"{subfolder_name}_clusters_{cluster_str}"
    out_dir_ts = Path(out_dir) / subfolder_name
    ensure_dir(out_dir_ts)

    # Create folder for bounding box txt files
    #bb_dir = out_dir_ts / "lat_log_bb"
    #ensure_dir(bb_dir)

    print(f"Saving images to: {out_dir_ts}")
    #print(f"Saving bounding boxes to: {bb_dir}")

    # Read GeoJSON
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.reset_index(drop=True)

    print(f"Max ID in local file: {gdf['id'].max()}")

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
    
    # Create folders for bounding boxes
    building_bb_dir = out_dir_ts / "building_bb"
    image_size_bb_dir = out_dir_ts / "image_size_bb"
    ensure_dir(building_bb_dir)
    ensure_dir(image_size_bb_dir)

    # inside your loop for each sampled polygon
    for i, row in sampled_3857.iterrows():
        try:
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Buffer polygon
            buffered = geom.buffer(buffer_m)

            # Convert buffered polygon to WGS84 (lon/lat)
            buffered_wgs = gpd.GeoSeries([buffered], crs=sampled_3857.crs).to_crs(epsg=4326).iloc[0]

            # Get requested (building) bounding box
            minx, miny, maxx, maxy = buffered_wgs.bounds
            pad_x = (maxx - minx) * 0.05
            pad_y = (maxy - miny) * 0.05
            west, south, east, north = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

            # Determine file name
            prop_id = row.get("id") if "id" in row.index else row.get("fid", i)
            name_fragment = f"{prop_id}"
            filename = Path(out_dir_ts) / f"{name_fragment}.png"

            # Save requested (building) bounding box
            req_bb_file = building_bb_dir / f"{name_fragment}.txt"
            with open(req_bb_file, "w") as f:
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

            # Convert true extent from Web Mercator to lat/lon

            extent_west, extent_east, extent_south, extent_north = ext
            poly_3857 = gpd.GeoSeries([box(extent_west, extent_south, extent_east, extent_north)], crs=3857)
            poly_wgs84 = poly_3857.to_crs(epsg=4326).iloc[0]
            true_west, true_south, true_east, true_north = poly_wgs84.bounds

            # Save true (image sized) bounding box in lat/lon
            image_size_bb_file = image_size_bb_dir / f"{name_fragment}.txt"
            with open(image_size_bb_file, "w") as f:
                f.write(f"{true_west},{true_south},{true_east},{true_north}")

            # Convert to RGB and save image
            if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
                img = np.transpose(img_arr, (1, 2, 0))
            else:
                img = img_arr
            pil_img = Image.fromarray(img)
            pil_img.save(filename)

            saved.append(str(filename))
            print(f"[{i}] saved image and bounding boxes ")

        except Exception as exc:
            print(f"[{i}] unexpected error: {exc}. Skipping.")

    if crop:
        crop_images_by_latlon(out_dir_ts,image_size_bb_dir,building_bb_dir) 

    # Return the list of files and allows the pipeline to continue to step two
    return saved, out_dir_ts


#------------#
# Execution
#------------#
if __name__ == "__main__":

    geojson_path = "/Users/willicon/Desktop/dronemodeling_Logan/buildingfootprint/Logan_utah.geojson" #"/Users/willicon/Desktop/dronemodeling_Logan/buildingfootprint/logan.geojson"
    out_dir = "/Users/willicon/Desktop"


    #Zoom 19 is closest zoom we can get
    #The seed tells what bulding to sample. Remebering the seed will allow it to be reproduced. 
    sample_and_save(geojson_path, 
                    out_dir, 
                    n=12, 
                    buffer_ft=50, 
                    zoom=19, 
                    seed=50,
                    crop = True)
                    #,clusters=[1,2,3]) # Clusters are given as list

