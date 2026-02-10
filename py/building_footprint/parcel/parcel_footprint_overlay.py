import geopandas as gpd
import folium
from folium.plugins import Search
import pandas as pd
import numpy as np
from shapely.ops import voronoi_diagram, unary_union
from shapely.geometry import MultiPoint, Polygon

# --- CONFIGURATION & TUNING ---

# 1. FILE PATHS
# ---------------------------------------------------------
NAME = "provo_edgemont_rock_canyon"
<<<<<<< HEAD
PARCEL_FILE = 'py/parcel/Parcels_Utah_-8019750519832152336.geojson'     #Geojson file for parcel dataset (Not included due to large file size) 
=======
PARCEL_FILE = 'py/parcel/Parcels_Utah_-8019750519832152336.geojson'      
>>>>>>> 42f1df5 (Added parcel method for building area capture.)
BUILDING_FILE = 'buildingfootprint/provo_orem_area/provo_edgemont_rock_canyon_kmeans_cluster.geojson' 

OUTPUT_PARCEL_FILE = f'buildingfootprint/parcel/{NAME}_parcels.geojson'
OUTPUT_HTML_FILE = f'buildingfootprint/parcel/{NAME}_viewer.html'

# The column in your building file that contains unique IDs (e.g., 'id', 'building_id')
BUILDING_ID_COLUMN = 'id' 


# 2. PARCEL GEOMETRY LOGIC
# ---------------------------------------------------------
# MERGE_SAME_ID_PARCELS (True/False):
#   - True: If a building spans 4 small parcels, and all 4 get assigned the same Building ID,
#     this will fuse them into ONE large parcel geometry.
#     (Best for: Apartment complexes, large commercial buildings on multiple lots).
#   - False: Keeps the 4 small parcels separate (resulting in 4 fragmented images).
MERGE_SAME_ID_PARCELS = True 

# SPLIT_PARCELS (True/False):
#   - True: Uses Voronoi diagrams to cut a single parcel into pieces if it contains multiple buildings.
#   - False: The entire parcel geometry is assigned to the building. If a parcel has 3 buildings,
#     the full parcel shape is duplicated 3 times in the output (once for each ID).
SPLIT_PARCELS = False   


# 3. BUILDING ASSIGNMENT LOGIC
# ---------------------------------------------------------
# ENFORCE_DOMINANT_PARCEL (True/False):
#   - True: "Winner Takes All." If a building sits on the border of Parcel A (60%) and Parcel B (40%),
#     it is assigned ONLY to Parcel A. Parcel B loses the building.
#     (Prevents duplicates where the same house appears in two different images).
#   - False: Both parcels claim the building.
ENFORCE_DOMINANT_PARCEL = True

# MIN_BUILDING_OVERLAP (0.0 to 1.0):
#   - The Standard Filter. A building must be at least X% inside the parcel to count.
#   - Example: 0.30 means 30%. If only 10% of a neighbor's house hangs over the fence 
#     into this parcel, it is ignored as a "sliver."
MIN_BUILDING_OVERLAP = 0.30 

# MIN_PARCEL_COVERAGE (0.0 to 1.0):
#   - The "Big Building" Rescue. 
#   - Sometimes a building is HUGE and the parcel is tiny. The overlap might only be 5% 
#     of the building, but it covers 100% of the parcel.
#   - Example: 0.50 means if the building covers 50% of the parcel's total area, 
#     we keep it, even if MIN_BUILDING_OVERLAP failed.
MIN_PARCEL_COVERAGE = 0.50


# 4. SHED / GARAGE REMOVAL
# ---------------------------------------------------------
# These parameters identify "Auxiliary Buildings" to remove them from the dataset entirely.

# GROUP_DISTANCE_THRESHOLD (Meters):
#   - Any building closer than this distance to a larger building is checked to see if it's a garage.
#   - Example: 15.0 meters.
GROUP_DISTANCE_THRESHOLD = 15.0  

# SPLIT_RATIO_THRESHOLD (0.0 to 1.0):
#   - If a building is within the distance threshold AND is smaller than this ratio 
#     of the main building, it is deleted.
#   - Example: 0.15 (15%). If the main house is 2000 sq ft, anything smaller than 300 sq ft 
#     nearby is considered a shed and removed.
SPLIT_RATIO_THRESHOLD = 0.15     

# ---------------------------------------------------------


def get_voronoi_regions(parcel_geom, seeds):
    """Generates Voronoi regions from seed points, clipped to the parcel."""
    if len(seeds) < 2:
        return [parcel_geom] * len(seeds)

    points = MultiPoint([s['centroid'] for s in seeds])
    try:
        envelope = parcel_geom.envelope.buffer(100) 
        regions = voronoi_diagram(points, envelope=envelope)
        
        ordered_regions = []
        for seed in seeds:
            pt = seed['centroid']
            match = None
            for region in regions.geoms:
                if region.contains(pt) or region.intersects(pt):
                    match = region
                    break
            if match:
                clipped = match.intersection(parcel_geom)
                ordered_regions.append(clipped)
            else:
                ordered_regions.append(None)
        return ordered_regions
    except Exception as e:
        print(f"Voronoi error: {e}")
        return [parcel_geom] * len(seeds)

def main():
    print("Loading data...")
    parcels = gpd.read_file(PARCEL_FILE)
    buildings = gpd.read_file(BUILDING_FILE)

    print("Projecting to EPSG:3857...")
    parcels_proj = parcels.to_crs(epsg=3857)
    buildings_proj = buildings.to_crs(epsg=3857)

    buildings_proj['centroid'] = buildings_proj.geometry.centroid
    buildings_proj['calc_area'] = buildings_proj.geometry.area

    print("Linking buildings to parcels...")
    buildings_with_parcel = gpd.sjoin(buildings_proj, parcels_proj, how="inner", predicate="intersects")

    # --- ENFORCE DOMINANT PARCEL ---
    if ENFORCE_DOMINANT_PARCEL:
        print("Calculating building dominance (Winner-Takes-All)...")
        keep_indices = []
        b_groups = buildings_with_parcel.groupby(BUILDING_ID_COLUMN)
        
        for b_id, b_subset in b_groups:
            if len(b_subset) == 1:
                keep_indices.append(b_subset.index[0])
                continue
            
            best_idx = None
            max_overlap = -1
            b_geom = b_subset.iloc[0]['geometry']
            
            for idx, row in b_subset.iterrows():
                p_geom = parcels_proj.loc[row['index_right']].geometry
                overlap = b_geom.intersection(p_geom).area
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_idx = idx
            
            if best_idx is not None:
                keep_indices.append(best_idx)
        
        print(f"Dropping {len(buildings_with_parcel) - len(keep_indices)} duplicate building references...")
        buildings_with_parcel = buildings_with_parcel.loc[keep_indices].copy()

    # -----------------------------------------

    new_geometries = []
    new_ids = []
    dropped_ids = [] 
    sliver_count = 0
    empty_parcel_count = 0
    rescued_count = 0

    print("Processing parcels...")
    grouped = buildings_with_parcel.groupby('index_right')
    
    for parcel_idx, subset in grouped:
        original_parcel = parcels_proj.loc[parcel_idx].geometry
        parcel_area = original_parcel.area
        
        b_data = []
        
        # --- STEP 1: SMART FILTER ---
        for idx, row in subset.iterrows():
            b_geom = row['geometry']
            b_area = row['calc_area']
            
            intersection = b_geom.intersection(original_parcel)
            overlap_area = intersection.area
            
            b_overlap_ratio = overlap_area / b_area
            p_coverage_ratio = overlap_area / parcel_area
            
            valid_match = False
            
            # Rule A: Does the parcel contain enough of the building?
            if b_overlap_ratio >= MIN_BUILDING_OVERLAP:
                valid_match = True
            # Rule B: Does the building cover enough of the parcel?
            elif p_coverage_ratio >= MIN_PARCEL_COVERAGE:
                valid_match = True
                rescued_count += 1
            
            if valid_match:
                b_data.append({
                    'id': row[BUILDING_ID_COLUMN],
                    'geometry': b_geom,
                    'centroid': row['centroid'],
                    'area': b_area
                })
            else:
                sliver_count += 1

        if not b_data:
            empty_parcel_count += 1
            continue

        # --- STEP 2: CLASSIFY SHEDS ---
        b_data.sort(key=lambda x: x['area'], reverse=True)
        
        groups = [] 
        assigned_ids = set()

        for b in b_data:
            if b['id'] in assigned_ids:
                continue
                
            current_group = {'main': b, 'extras': []}
            assigned_ids.add(b['id'])
            
            for candidate in b_data:
                if candidate['id'] in assigned_ids:
                    continue
                
                dist = b['centroid'].distance(candidate['centroid'])
                is_close = dist < GROUP_DISTANCE_THRESHOLD
                is_small = candidate['area'] < (b['area'] * SPLIT_RATIO_THRESHOLD)
                
                if is_close or is_small:
                    current_group['extras'].append(candidate)
                    assigned_ids.add(candidate['id'])
                    dropped_ids.append(candidate['id']) 
            
            groups.append(current_group)

        # --- STEP 3: ASSIGN GEOMETRY ---
        if not SPLIT_PARCELS:
            for g in groups:
                new_geometries.append(original_parcel)
                new_ids.append(g['main']['id'])
            continue 

        # Branch: Splitting (Voronoi)
        seeds = [g['main'] for g in groups]

        if len(seeds) == 1:
            regions = [original_parcel]
        else:
            regions = get_voronoi_regions(original_parcel, seeds)

        # Geometric Repair
        all_mains_geom = [g['main']['geometry'] for g in groups]

        for i, group_data in enumerate(groups):
            base_region = regions[i]
            if base_region is None: continue

            my_main_geom = group_data['main']['geometry']
            other_mains = [geom for geom in all_mains_geom if geom != my_main_geom]
            
            if other_mains:
                other_shape_union = unary_union(other_mains)
                final_shape = base_region.difference(other_shape_union).union(my_main_geom)
            else:
                final_shape = base_region.union(my_main_geom)

            if not final_shape.is_empty:
                new_geometries.append(final_shape)
                new_ids.append(group_data['main']['id'])

    # --- COMPILE GDF ---
    final_gdf = gpd.GeoDataFrame({
        BUILDING_ID_COLUMN: new_ids,
        'geometry': new_geometries
    }, crs=parcels_proj.crs)

    # --- NEW STEP: MERGE SAME ID PARCELS ---
    if MERGE_SAME_ID_PARCELS and not final_gdf.empty:
        print("Merging fragmented parcels (Dissolving by Building ID)...")
        initial_count = len(final_gdf)
        
        # 'dissolve' combines geometries for rows with the same ID
        final_gdf = final_gdf.dissolve(by=BUILDING_ID_COLUMN, as_index=False)
        
        print(f"Merged {initial_count} fragments into {len(final_gdf)} consolidated parcels.")

    # --- SAVE OUTPUT ---
    final_output = final_gdf.to_crs(epsg=4326)
    
    if not final_output.empty:
        final_output.to_file(OUTPUT_PARCEL_FILE, driver='GeoJSON')
        print(f"Saved to {OUTPUT_PARCEL_FILE}")
        
        # --- MAP GENERATION ---
        print("Generating Map...")
        bounds = final_output.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="CartoDB positron")

        parcel_layer = folium.GeoJson(
            final_output,
            name="Smart Parcels",
            style_function=lambda x: {'fillColor': 'green', 'color': 'white', 'weight': 1, 'fillOpacity': 0.4},
            tooltip=folium.GeoJsonTooltip(fields=[BUILDING_ID_COLUMN], aliases=['ID:']),
            popup=folium.GeoJsonPopup(fields=[BUILDING_ID_COLUMN], aliases=['Building ID:'])
        ).add_to(m)
        
        Search(
            layer=parcel_layer,
            geom_type='Polygon',
            placeholder="Search for Building ID...",
            collapsed=False,
            search_label=BUILDING_ID_COLUMN,
            weight=3
        ).add_to(m)
        
        clean_buildings_gdf = buildings_proj[~buildings_proj[BUILDING_ID_COLUMN].isin(dropped_ids)].copy()
        clean_buildings_gdf = clean_buildings_gdf.drop(columns=['centroid', 'calc_area']).to_crs(epsg=4326)
        
        folium.GeoJson(
            clean_buildings_gdf,
            name="Main Buildings Only",
            style_function=lambda x: {'color': 'black', 'weight': 1, 'fillOpacity': 0.6},
            tooltip="Main Building"
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        m.save(OUTPUT_HTML_FILE)
        print(f"Map saved to {OUTPUT_HTML_FILE}")
        
    else:
        print("No output generated.")

if __name__ == "__main__":
    main()