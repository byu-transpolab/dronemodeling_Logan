import pandas as pd
import geopandas as gpd
from shapely import geometry
import mercantile
from tqdm import tqdm
import os
import tempfile
import folium

import webbrowser
from libpysal.weights import Queen
from spopt.region import MaxPHeuristic
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def view_geojson(geojson_path):
    # Load the GeoJSON with GeoPandas
    gdf = gpd.read_file(geojson_path)

    # Reproject to a projected CRS for accurate centroid
    gdf_proj = gdf.to_crs(epsg=3857)

    # Compute centroid on projected coordinates, then transform back to lat/lon
    centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    center = [centroid.y.mean(), centroid.x.mean()]

    # Create Folium map centered on the data
    m = folium.Map(location=center, zoom_start=13)

    # Add the GeoJSON layer to the map
    folium.GeoJson(geojson_path, name="GeoJSON Layer").add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Remove the original file extension and add .html
    base, _ = os.path.splitext(geojson_path)
    output_html = base + ".html"

    # Save to HTML
    m.save(output_html)

    print(f"Map saved to {output_html}")


    #Open in web browser
    print("Opening in browser")

    # Path to the saved HTML file
    geojsonbase, _ = os.path.splitext(geojson_path)

    # Add .html
    output_html = geojsonbase + ".html"
   
    # Convert to absolute path
    full_path = os.path.abspath(output_html)#output_html

    webbrowser.open(f"file://{full_path}", new=2)
    print(f"file://{full_path}")



# This script will pull the building footprint data from the microsoft building footprint data set.
def pull_building_footprint (area_coordinates,output_folder):
    #Define Area of Intrest
  
  aoi_geom = {
      "coordinates": [
          area_coordinates
      ],
      "type": "Polygon",
  }

  aoi_shape = geometry.shape(aoi_geom)
  minx, miny, maxx, maxy = aoi_shape.bounds

  #Intersecting tiles
  quad_keys = set()
  for tile in list(mercantile.tiles(minx, miny, maxx, maxy, zooms=9)):
      quad_keys.add(mercantile.quadkey(tile))
  quad_keys = list(quad_keys)
  print(f"The input area spans {len(quad_keys)} tiles: {quad_keys}")

  #Download the data
  df = pd.read_csv(
      "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv", dtype=str
  )
  df.head()

  print("hi")

  idx = 0
  combined_gdf = gpd.GeoDataFrame()
  with tempfile.TemporaryDirectory() as tmpdir:
      # Download the GeoJSON files for each tile that intersects the input geometry
      tmp_fns = []
      for quad_key in tqdm(quad_keys):
          rows = df[df["QuadKey"] == quad_key]
          if rows.shape[0] == 1:
              url = rows.iloc[0]["Url"]

              df2 = pd.read_json(url, lines=True)
              df2["geometry"] = df2["geometry"].apply(geometry.shape)

              gdf = gpd.GeoDataFrame(df2, crs=4326)
              fn = os.path.join(tmpdir, f"{quad_key}.geojson")
              tmp_fns.append(fn)
              if not os.path.exists(fn):
                  gdf.to_file(fn, driver="GeoJSON")
          elif rows.shape[0] > 1:
              raise ValueError(f"Multiple rows found for QuadKey: {quad_key}")
          else:
              raise ValueError(f"QuadKey not found in dataset: {quad_key}")

      # Merge the GeoJSON files into a single file
      for fn in tmp_fns:
          gdf = gpd.read_file(fn)  # Read each file into a GeoDataFrame
          gdf = gdf[gdf.geometry.within(aoi_shape)]  # Filter geometries within the AOI
          gdf['id'] = range(idx, idx + len(gdf))  # Update 'id' based on idx
          idx += len(gdf)
          combined_gdf = pd.concat([combined_gdf,gdf],ignore_index=True)

    #Save to file        
  combined_gdf = combined_gdf.to_crs('EPSG:4326')
  combined_gdf.to_file(output_folder, driver='GeoJSON')



def Spatially_constrained_clustering(geojson_file):

    gdf = gpd.read_file(geojson_file)

    # Spatial weights (adjacency)
    w = Queen.from_dataframe(gdf)

    # Attribute for clustering (e.g., population)
    attrs = gdf[['population']].values

    # Max-p clustering
    model = MaxPHeuristic(gdf, w, attrs, threshold=1000, top_n=5)
    model.solve()

    gdf['cluster'] = model.labels_
    gdf.plot(column='cluster', categorical=True, legend=True)
'''
def k_mean_clustering(geojson_file):
    gdf = gpd.read_file(geojson_file)

    # Extract centroids
    gdf["centroid"] = gdf.geometry.centroid
    coords = np.vstack((gdf.centroid.x, gdf.centroid.y)).T

    # Cluster with DBSCAN
    db = DBSCAN(eps=10, min_samples=5).fit(coords)  # tune eps depending on your data
    gdf["cluster_id"] = db.labels_

    # Plot results
    gdf.plot(column="cluster_id", categorical=True, legend=True)

def view_cluster(geojson_file):
# Convert cluster labels to colors or icons
    gdf = gpd.read_file(geojson_file)
    m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=10)
    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in gdf.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"Cluster: {row['cluster']}"
        ).add_to(marker_cluster)

    m.save("cluster_map.html")
'''
def k_value_elbow_method(X_scaled):
    # include k=1 in the range
    K = range(1, 20)

    inertia = []
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    # normalize inertia relative to k=1
    baseline = inertia[0]
    inertia_normalized = [i / baseline for i in inertia]

    plt.figure(figsize=(8,5))
    plt.plot(list(K), inertia_normalized, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Relative Inertia (normalized to k=1)')
    plt.title('Elbow Method — Normalized Inertia')
    plt.grid(True)
    plt.show()

def k_mean_prepare_area_perimeter_scaled(geojson_file):

    gdf = gpd.read_file(geojson_file)

    # Project to a CRS with meters 
    gdf = gdf.to_crs(epsg=3857)  # or use your local UTM zone

    #Define area and perimeter
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length

    # Create feature matrix
    X = gdf[["area", "perimeter"]].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return (gdf, X_scaled)

def k_mean_analysis(gdf, X_scaled, k_value, view):
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init="auto")
    gdf["cluster"] = kmeans.fit_predict(X_scaled)

    if view == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column="cluster", categorical=True, legend=True, ax=ax)
        ax.set_title("K-Means Clustering: Area vs Perimeter")
        plt.show()

    elif view == 2:
        plt.figure(figsize=(10,7))
        scatter = plt.scatter(
            gdf["area"],
            gdf["perimeter"],
            c=gdf["cluster"],
            cmap="tab10",
            alpha=0.7
        )
        plt.xlabel("Area (m²)")
        plt.ylabel("Perimeter (m)")
        plt.title("K-Means Clustering of Buildings by Area vs Perimeter")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.show()


geojson_file = "buildingfootprint/logan_building_footprints.geojson"

gdf, X_scaled = k_mean_prepare_area_perimeter_scaled(geojson_file)

#print(X_scaled)

#k_value_elbow_method(X_scaled)

k_value = 8

k_mean_analysis(gdf, X_scaled,k_value,2)