import geopandas as gpd
import contextily as ctx
from PIL import Image, ImageDraw
import numpy as np
import os
from pathlib import Path
import random
from datetime import datetime
from shapely.geometry import box

# --- CONFIGURATION ---
PARCEL_FILE = 'buildingfootprint/parcel/provo_edgemont_rock_canyon_parcels.geojson'
BUILDING_FILE = 'buildingfootprint/provo_orem_area/provo_edgemont_rock_canyon_kmeans_cluster.geojson'
OUT_DIR = "/Users/willicon/Desktop"

ZOOM_LEVEL = 19
SAMPLES = 50
SEED = 15

# --- TUNING PARAMETERS ---

# 1. RATIO THRESHOLD
# If Parcel Area / Building Area < 1.05, switch to Buffer (option B)
AREA_RATIO_THRESHOLD = 1.05  

# 2. NEIGHBOR FILTER 
# To count as a valid neighbor (triggering Option B), a building must be:
# A. OVERLAP: At least "NEIGHBOR_OVERLAP_THRESHOLD"% inside the parcel.
NEIGHBOR_OVERLAP_THRESHOLD = 0.15 
# B. SIZE: At least "neighbor_size ratio" % the size of the Main Building (Ignore sheds).
NEIGHBOR_SIZE_RATIO = 0.25

# 3. OPTION A (MASK) SETTINGS
#Give a buffer around the parcel area
MASK_BUFFER_FT = 10.0   

# 4. OPTION B (BUFFER) SETTINGS
BUFFER_DISTANCE_FT = 50     
BUFFER_WHOLE_GROUP = False 
#If true, will capture parcels with multiple buildings togther
# ---------------------

FEET_TO_METERS = 0.3048

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def world_to_pixel(x, y, bounds, img_width, img_height):
    minx, miny, maxx, maxy = bounds
    x_ratio = (x - minx) / (maxx - minx)
    y_ratio = (y - miny) / (maxy - miny)
    px = int(x_ratio * img_width)
    py = int((1 - y_ratio) * img_height) 
    return px, py

# --- METHOD A: PARCEL MASKING ---
def method_mask_parcel(geom, out_path, file_id, zoom, buffer_ft):
    if buffer_ft > 0:
        buffer_m = buffer_ft * FEET_TO_METERS
        geom = geom.buffer(buffer_m)

    minx, miny, maxx, maxy = geom.bounds
    try:
        img_arr, ext = ctx.bounds2img(minx, miny, maxx, maxy, zoom=zoom, ll=False, source=ctx.providers.Esri.WorldImagery)
    except Exception as e:
        print(f"  [Mask] Tile fetch failed: {e}")
        return False

    image_extent = (ext[0], ext[2], ext[1], ext[3]) 
    
    if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
        img_arr = np.transpose(img_arr, (1, 2, 0))
    
    source_img = Image.fromarray(img_arr).convert("RGBA")
    width, height = source_img.size

    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
    
    for g in geoms:
        exterior = list(g.exterior.coords)
        pixels = [world_to_pixel(x, y, image_extent, width, height) for x, y in exterior]
        draw.polygon(pixels, fill=255)
        for interior in g.interiors:
            inner = list(interior.coords)
            inner_px = [world_to_pixel(x, y, image_extent, width, height) for x, y in inner]
            draw.polygon(inner_px, fill=0)

    source_img.putalpha(mask)
    
    alpha = source_img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        final_img = source_img.crop(bbox)
        final_img.save(out_path / f"{file_id}.png")
        return True
    return False

# --- METHOD B: BUFFER & CROP ---
def method_buffer_building(building_geoms, out_path, bb_dir, size_dir, file_id, zoom, buffer_ft):
    gs = gpd.GeoSeries(building_geoms)
    if hasattr(gs, 'union_all'):
        combined_geom = gs.union_all() 
    else:
        combined_geom = gs.unary_union
    
    buffer_m = buffer_ft * FEET_TO_METERS
    buffered_geom_3857 = combined_geom.buffer(buffer_m)
    buffered_geom_4326 = gpd.GeoSeries([buffered_geom_3857], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
    
    minx, miny, maxx, maxy = buffered_geom_4326.bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    west, south, east, north = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)

    req_bb_file = bb_dir / f"{file_id}.txt"
    with open(req_bb_file, "w") as f:
        f.write(f"{west},{south},{east},{north}")

    img_arr = None
    ext = None
    success = False
    
    current_zoom = zoom
    while current_zoom >= 10:
        try:
            img_arr, ext = ctx.bounds2img(west, south, east, north, zoom=current_zoom, ll=True, source=ctx.providers.Esri.WorldImagery)
            success = True
            break
        except Exception:
            current_zoom -= 1
            
    if not success:
        print(f"  [Buffer] Failed to fetch tiles for ID {file_id}")
        return False

    extent_west, extent_east, extent_south, extent_north = ext
    poly_3857 = gpd.GeoSeries([box(extent_west, extent_south, extent_east, extent_north)], crs=3857)
    poly_wgs84 = poly_3857.to_crs(epsg=4326).iloc[0]
    true_west, true_south, true_east, true_north = poly_wgs84.bounds
    
    image_size_bb_file = size_dir / f"{file_id}.txt"
    with open(image_size_bb_file, "w") as f:
        f.write(f"{true_west},{true_south},{true_east},{true_north}")

    if img_arr.ndim == 3 and img_arr.shape[0] in (3, 4):
        img_arr = np.transpose(img_arr, (1, 2, 0))
    pil_img = Image.fromarray(img_arr)
    width, height = pil_img.size

    def lon_to_px(lon):
        return int(round((lon - true_west) / (true_east - true_west) * width))
    def lat_to_py(lat):
        return int(round((true_north - lat) / (true_north - true_south) * height))

    left = lon_to_px(west)
    upper = lat_to_py(north)
    right = lon_to_px(east)
    lower = lat_to_py(south)

    crop_box = (max(0, left), max(0, upper), min(width, right), min(height, lower))
    
    try:
        cropped_img = pil_img.crop(crop_box)
        cropped_img.save(out_path / f"{file_id}.png")
        return True
    except Exception as e:
        print(f"  [Buffer] Crop error ID {file_id}: {e}")
        return False

def main():
    location = Path(PARCEL_FILE).stem
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    out_path = Path(OUT_DIR) / f"{location}_building_driven_seed{SEED}_{timestamp}"
    ensure_dir(out_path)
    
    bb_dir = out_path / "building_bb"
    size_dir = out_path / "image_size_bb"
    ensure_dir(bb_dir)
    ensure_dir(size_dir)
    
    print(f"Saving to: {out_path}")

    print("Loading data...")
    parcels = gpd.read_file(PARCEL_FILE).to_crs(epsg=3857)
    buildings = gpd.read_file(BUILDING_FILE).to_crs(epsg=3857)
    
    parcels['id'] = parcels['id'].astype(str)
    buildings['id'] = buildings['id'].astype(str)
    
    parcels = parcels[parcels.geometry.is_valid]
    buildings = buildings[buildings.geometry.is_valid]
    
    if SEED: random.seed(SEED)
    sample_size = min(SAMPLES, len(buildings))
    sampled_buildings = buildings.sample(n=sample_size, random_state=SEED)
    
    print(f"Sampled {sample_size} buildings for processing...")
    
    count_mask = 0
    count_buffer = 0
    
    for idx, b_row in sampled_buildings.iterrows():
        b_id = b_row['id']
        b_geom = b_row.geometry
        b_area = b_geom.area
        
        # 1. FIND ASSOCIATED PARCEL
        associated_parcel = parcels[parcels.contains(b_geom.centroid)].copy()
        
        if associated_parcel.empty:
            associated_parcel = parcels[parcels.intersects(b_geom)].copy()
            if not associated_parcel.empty:
                 associated_parcel['overlap'] = associated_parcel.geometry.intersection(b_geom).area
                 associated_parcel = associated_parcel.sort_values('overlap', ascending=False).head(1)
        
        if associated_parcel.empty:
            print(f"ID {b_id}: No parcel found. Defaulting to Buffer.")
            success = method_buffer_building([b_geom], out_path, bb_dir, size_dir, b_id, ZOOM_LEVEL, BUFFER_DISTANCE_FT)
            if success: count_buffer += 1
            continue
            
        p_row = associated_parcel.iloc[0]
        p_geom = p_row.geometry
        
        # 2. CONTEXT CHECK (SMART NEIGHBOR FILTER)
        raw_neighbors = buildings[buildings.geometry.intersects(p_geom)]
        
        valid_neighbors = []
        debug_info = []

        for n_idx, n_row in raw_neighbors.iterrows():
            n_id = n_row['id']
            # Ignore self
            if n_id == b_id:
                continue
            
            inter_area = n_row.geometry.intersection(p_geom).area
            n_area = n_row.geometry.area
            overlap_pct = inter_area / n_area
            size_ratio = n_area / b_area
            
            # Debugging string
            status = "Ignored"
            
            # Check 1: Is it a Sliver?
            if overlap_pct > NEIGHBOR_OVERLAP_THRESHOLD:
                # Check 2: Is it a Shed?
                if size_ratio > NEIGHBOR_SIZE_RATIO:
                    valid_neighbors.append(n_row)
                    status = "VALID"
                else:
                    status = "Ignored (Small Shed)"
            else:
                status = "Ignored (Sliver)"
            
            debug_info.append(f"[N:{n_id} Overlap:{overlap_pct:.2f} Size:{size_ratio:.2f} -> {status}]")
        
        # Add the main building to the count for ratio calc
        if valid_neighbors:
            valid_gdf = gpd.GeoDataFrame(valid_neighbors, crs=buildings.crs)
            total_b_area = valid_gdf.geometry.area.sum() + b_area
            num_extra_buildings = len(valid_neighbors)
        else:
            total_b_area = b_area
            num_extra_buildings = 0

        p_area = p_geom.area
        ratio = p_area / total_b_area if total_b_area > 0 else 999
        
        use_buffer_method = False
        reason = ""
        
        # Decision Logic
        if num_extra_buildings > 0:
            use_buffer_method = True
            reason = f"Multi-Building Found! {debug_info}"
        elif ratio < AREA_RATIO_THRESHOLD:
            use_buffer_method = True
            reason = f"Tight Ratio ({ratio:.2f})"
        else:
            use_buffer_method = False
            reason = f"Standard (Ratio {ratio:.2f})"
            
        # DEBUG PRINT
        print(f"Processing ID {b_id}: {reason}")

        # 3. EXECUTION
        success = False
        if use_buffer_method:
            if BUFFER_WHOLE_GROUP and valid_neighbors:
                # Include neighbors in buffer if requested
                group_geom = valid_gdf.geometry.tolist() + [b_geom]
            else:
                group_geom = [b_geom]
            
            success = method_buffer_building(group_geom, out_path, bb_dir, size_dir, b_id, ZOOM_LEVEL, BUFFER_DISTANCE_FT)
            if success: count_buffer += 1
        else:
            success = method_mask_parcel(p_geom, out_path, b_id, ZOOM_LEVEL, MASK_BUFFER_FT)
            if success: count_mask += 1

    print("Done!")
    print(f"Total Masked (Option A): {count_mask}")
    print(f"Total Buffered (Option B): {count_buffer}")

if __name__ == "__main__":
    main()