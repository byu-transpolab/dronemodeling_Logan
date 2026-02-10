import geopandas as gpd
import numpy as np
"""
Will reproduce the lat/long coord. used to create your geojson files in the buliding area code.
"""


def get_geojson_polygon_extent(geojson_path):
    print(f"Reading {geojson_path}...")
    try:
        # 1. Load the file
        gdf = gpd.read_file(geojson_path)
        
        # 2. Ensure Lat/Lon (EPSG:4326)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        print(f"Processing {len(gdf)} buildings to find tightest polygon...")

        # 3. Create a Convex Hull (The "rubber band" around all buildings)
        # We use unary_union to treat all buildings as one object, then get the hull
        combined_geom = gdf.geometry.unary_union
        hull = combined_geom.convex_hull

        # 4. Extract coordinates
        if hull.geom_type == 'Polygon':
            # Get the exterior coordinates
            coords = list(hull.exterior.coords)
        else:
            # Fallback if the hull is somehow a line or point (unlikely)
            print("Geometry is not a polygon (it might be a line or point).")
            return

        # 5. Print formatted code for your script
        print("\n" + "="*40)
        print("COPY AND PASTE THIS INTO 'custom_coords':")
        print("="*40)
        print("custom_coords = [")
        
        for lon, lat in coords:
            # Rounding to 6 decimal places for cleanliness
            print(f"    [{round(lon, 6)}, {round(lat, 6)}],")
            
        print("]")
        print("="*40)

    except Exception as e:
        print(f"Error: {e}")

# ---- USAGE ----
# Replace with your file path
input_file = "/Users/willicon/Desktop/dronemodeling_Logan/buildingfootprint/provo_orem_area/vinyard.geojson" 

get_geojson_polygon_extent(input_file)