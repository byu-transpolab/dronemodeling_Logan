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
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors

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
def pull_building_footprint (area_coordinates,output_folder_path,name):
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
  combined_gdf.to_file(f"{output_folder_path}/{name}.geojson", driver='GeoJSON')



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


def k_value_elbow_method(X_scaled, plot=True, k_min=1, k_max=20):
    K = range(k_min, k_max + 1)
    inertia = []

    # Compute inertia for each k
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)

    # Normalize inertia relative to k=1 for better visualization
    baseline = inertia[0]
    inertia_normalized = [i / baseline for i in inertia]

    # Find the "elbow" point automatically
    kn = KneeLocator(
        K, inertia_normalized, curve="convex", direction="decreasing"
    )
    best_k = kn.knee

    # Plot the elbow curve
    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(K, inertia_normalized, marker='o')
        if best_k is not None:
            plt.axvline(best_k, color='r', linestyle='--', label=f"Best k = {best_k}")
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Relative Inertia (normalized to k=1)')
        plt.title('Elbow Method — Normalized Inertia')
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Optimal number of clusters (k): {best_k}")
    return best_k


def k_mean_prepare_area_perimeter_scaled(geojson_file):

    gdf = gpd.read_file(geojson_file)

    # Project to a CRS (meters)
    gdf = gdf.to_crs(epsg=3857)  # local UTM zone?

    #Define area and perimeter
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length

    # Create feature matrix
    X = gdf[["area", "perimeter"]].values

    # Normalize scale to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return (gdf, X_scaled)

def k_mean_analysis(gdf, X_scaled, k_value, view):
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init="auto")
    gdf["cluster"] = kmeans.fit_predict(X_scaled)

    #Shows building custers while maintaining thier locations
    if view == 1:
        print("Ploting building relative to their location")
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column="cluster", categorical=True, legend=True, ax=ax)
        ax.set_title("K-Means Clustering: Area vs Perimeter")
        plt.show()

    #Shows clusters on normalized plot
    elif view == 2:
        print("Ploting clusters on normalized graph")
        plt.figure(figsize=(10,7))
        scatter = plt.scatter(
            X_scaled[:, 0],  # normalized area
            X_scaled[:, 1],  # normalized perimeter
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
    elif view ==3:
        print("No plot. Returning geojson_file with cluster information ")

    return gdf    


def view_geojson_with_clusters(gdf, geojson_path):
    print("Creating html map with cluster data")

    # --- Ensure we're in a projected CRS before computing centroids ---
    gdf_proj = gdf.to_crs(epsg=3857)  # Projected (meters)
    centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    center = [centroid.y.mean(), centroid.x.mean()]

    # --- Convert to WGS84 for Folium ---
    gdf = gdf.to_crs(epsg=4326)

    # --- Create Folium map ---
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")

    # --- Create color mapping for clusters ---
    #n_clusters = gdf["cluster"].nunique()
    cmap = matplotlib.colormaps.get_cmap("tab10")  # for Matplotlib >= 3.7
    norm = colors.Normalize(vmin=gdf["cluster"].min(), vmax=gdf["cluster"].max())

    def style_function(feature):
        cluster_id = feature["properties"]["cluster"]
        rgba = cmap(norm(cluster_id))
        color = matplotlib.colors.rgb2hex(rgba)
        return {
            "fillColor": color,
            "color": color,
            "weight": 1,
            "fillOpacity": 0.6,
        }

    # --- Add polygons to map ---
    folium.GeoJson(
        data=gdf.to_json(),
        name="Building Clusters",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=["cluster"]),
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # --- Save to HTML ---
    base, _ = os.path.splitext(geojson_path)
    output_html = base + "_clusters.html"
    m.save(output_html)

    print(f"\nCluster map saved to: {output_html}")
    full_path = os.path.abspath(output_html)

    return full_path

#This will run the whole process to create a html file of the kmeans analysis. 
def run_kmeans_cluster_view(geojson_path,view=3):
    gdf, X_scaled = k_mean_prepare_area_perimeter_scaled(geojson_path)
    k_value = k_value_elbow_method(X_scaled,False)
    gdf = k_mean_analysis(gdf, X_scaled, k_value, view=view)
    html_path = view_geojson_with_clusters(gdf, geojson_path)
    return html_path        





#Define Path and name
output_folder_path = "buildingfootprint"
outputname = "logan"


#Define Area of intrerest 
# Geometry copied from https://geojson.io ----Current Coordinates are for Logan, Utah
area_coordinates_logan = [
              [-111.87734176861632,
                41.809086498918845],
              [-111.87734176861632,
                41.66579543007279],
              [-111.7753058035783,
                41.66579543007279],
              [-111.7753058035783,
                41.809086498918845],
              [-111.87734176861632,
                41.809086498918845]
            ]

area_coordinates_roanoke = [
            [
              -80.14070196114528,
              37.366555645142654
            ],
            [
              -80.14070196114528,
              37.19513563929465
            ],
            [
              -79.80639472292154,
              37.19513563929465
            ],
            [
              -79.80639472292154,
              37.366555645142654
            ],
            [
              -80.14070196114528,
              37.366555645142654
            ]
          ]


pull_building_footprint(area_coordinates_logan,output_folder_path,outputname)


geojson_file = f"{output_folder_path}/{outputname}.geojson"

run_kmeans_cluster_view(geojson_file,2)
