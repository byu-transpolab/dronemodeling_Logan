import pandas as pd
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats

def process_building_data(geojson_path, csv_path, k=5, geojson_id='id', csv_id='ID'):
    """
    Merges GeoJSON spatial data with YOLO CSV data and generates 
    neighbor/cluster context variables.
    
    Args:
        geojson_path: Path to the .geojson file
        csv_path: Path to the .csv file
        k: Number of nearest neighbors to consider (default=5)
        geojson_id: The ID column name in the geojson. Needed to combine the two files. 
        csv_id: The ID column name in the csv
    """
    
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print(f"Loading CSV: {csv_path}")
    df_attributes = pd.read_csv(csv_path)
    
    print(f"Loading GeoJSON: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    
    # ---------------------------------------------------------
    # 2. STANDARDIZE & MERGE
    # ---------------------------------------------------------
    # Convert ID columns to strings
    gdf[geojson_id] = gdf[geojson_id].astype(str)
    df_attributes[csv_id] = df_attributes[csv_id].astype(str)
    
    print(f"Merging files...")
    full_df = gdf.merge(df_attributes, left_on=geojson_id, right_on=csv_id, how='inner')
    print(f"Merged Data: {len(full_df)} buildings matched.")

    # ---------------------------------------------------------
    # 3. FIX COORDINATE SYSTEM
    # ---------------------------------------------------------
    # We estimate the best UTM projection (meters) for this specific location.
    # This makes distance calculations accurate
    print("Projecting to UTM (Meters) for accurate calculations...")
    
    # Check if CRS is already projected, if not, estimate one
    if not full_df.crs.is_projected:
        utm_crs = full_df.estimate_utm_crs()
        full_df_projected = full_df.to_crs(utm_crs)
    else:
        full_df_projected = full_df

    # Extract Centroids (in meters)
    # We use these projected coordinates for the Nearest Neighbor math
    full_df_projected['centroid_x'] = full_df_projected.geometry.centroid.x
    full_df_projected['centroid_y'] = full_df_projected.geometry.centroid.y
    
    # ---------------------------------------------------------
    # 4. GENERATE NEIGHBOR FEATURES (Spatial Lag)
    # ---------------------------------------------------------
    print(f"Calculating Nearest Neighbor features (k={k})...")
    
    coords = full_df_projected[['centroid_x', 'centroid_y']].values
    
    # Fit Nearest Neighbors algorithm
    knn = NearestNeighbors(n_neighbors=k+1) # k+1 because the building finds itself
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)
    
    # Lists to store the new variables
    neighbor_mean_area = []
    neighbor_mean_perimeter = [] 
    neighbor_mean_conf = []
    neighbor_majority_class = []
    
    for i, neighbor_indices in enumerate(indices):
        real_neighbors_idx = neighbor_indices[1:] # Exclude the building itself
        neighbor_data = full_df_projected.iloc[real_neighbors_idx]
        
        # --- CALCULATE STATISTICS ---
        neighbor_mean_area.append(neighbor_data['Area_sqm'].mean())
        neighbor_mean_perimeter.append(neighbor_data['Perimeter_m'].mean())
        neighbor_mean_conf.append(neighbor_data['Confidence'].mean())
        
        # Mode (Most common YOLO prediction)
        mode_res = stats.mode(neighbor_data['yolo_pred'], keepdims=True)
        neighbor_majority_class.append(mode_res.mode[0])

    # Assign new columns
    # We assign them back to the original (or projected) dataframe
    full_df_projected['neighbor_mean_area'] = neighbor_mean_area
    full_df_projected['neighbor_mean_perimeter'] = neighbor_mean_perimeter
    full_df_projected['neighbor_mean_conf'] = neighbor_mean_conf
    full_df_projected['neighbor_majority_class'] = neighbor_majority_class

    # ---------------------------------------------------------
    # 5. GENERATE CLUSTER FEATURES
    # ---------------------------------------------------------
    print("Generating Cluster features...")
    
    # Use Centroids(Meters) + Area + Perimeter
    features_to_cluster = full_df_projected[['centroid_x', 'centroid_y', 'Area_sqm', 'Perimeter_m']].copy()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_cluster)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    full_df_projected['spatial_cluster_id'] = kmeans.fit_predict(scaled_features)

    # ---------------------------------------------------------
    # 6. OUTPUT SETUP
    # ---------------------------------------------------------
    # Clean up geometry for CSV export
    df_final = pd.DataFrame(full_df_projected.drop(columns='geometry'))
    
    # Remove extra ID column if exists
    if csv_id in df_final.columns and csv_id != geojson_id:
        df_final = df_final.drop(columns=[csv_id])

    # Determine Output Path
    # Logic: Get CSV path -> Parent -> Parent (Grandparent) -> New Folder
    csv_path_obj = Path(csv_path)
    grandparent_folder = csv_path_obj.parent.parent
    output_folder = grandparent_folder / "neighbor calculations"
    
    # Create folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"spatial_features_k{k}.csv"
    output_path = output_folder / output_filename
    
    print(f"Saving results to: {output_path}")
    df_final.to_csv(output_path, index=False)
    
    return df_final

# ==========================================
# EXECUTION
# ==========================================

# 1. Define your file paths here
geojson_file = 'buildingfootprint/provo_orem_area/orem_caschade_orchard.geojson'
csv_file = 'output_data/orem_caschade_orchard_seed15_2026_02_04_083646/yolo_pred/yolo_output_data.csv'

# 2. Run the function with k as an input
final_df = process_building_data(
    geojson_file, 
    csv_file, 
    k=5,              # <--- CHANGE K HERE
    geojson_id='id', 
    csv_id='ID'
)

# 3. View and Save
print("\nSuccess! Sample of the new variables:")
cols_to_show = ['id', 'yolo_pred', 'neighbor_majority_class', 'neighbor_mean_area']
print(final_df[cols_to_show].head())