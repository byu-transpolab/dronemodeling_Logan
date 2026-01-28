import pandas as pd
import geopandas as gpd
from shapely import geometry
import mercantile
from tqdm import tqdm
import os
import tempfile
import folium

from geopy.geocoders import Nominatim
import webbrowser
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from pathlib import Path
import random

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

def get_city_border_coordinates(city_name):
    print(f"Fetching boundary data for: {city_name}...")
    geolocator = Nominatim(user_agent="my_building_analysis_app")
    
    # Request the geometry ('geojson') to get the full shape
    location = geolocator.geocode(city_name, geometry='geojson')
    
    if not location:
        raise ValueError(f"City '{city_name}' not found.")

    geojson = location.raw.get('geojson')
    
    if not geojson or 'coordinates' not in geojson:
        raise ValueError("No boundary data found for this location.")

    geo_type = geojson.get('type')
    coords = geojson.get('coordinates')
    
    # Return the outer list of coordinates [ [lon, lat], [lon, lat]... ]
    if geo_type == 'Polygon':
        return coords[0]
    elif geo_type == 'MultiPolygon':
        # For MultiPolygons, we take the first polygon (usually the main city area)
        return coords[0][0]
    else:
        raise ValueError(f"Geometry type {geo_type} not supported.")

def get_custom_box_coordinates(west, south, east, north):
    """
    Creates a rectangular polygon from bounding box coordinates.
    Returns a list of [lon, lat] points closing the loop.
    """
    print(f"Using custom bounding box: {west}, {south}, {east}, {north}")
    return [
        [west, south],  # Bottom-Left
        [east, south],  # Bottom-Right
        [east, north],  # Top-Right
        [west, north],  # Top-Left
        [west, south]   # Close the loop
    ]
# This script will pull the building footprint data from the microsoft building footprint data set.
def pull_building_footprints (area_coordinates,output_folder_path,name):
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
          
          gdf['id'] = range(idx, idx + len(gdf))  
          
          idx += len(gdf)
          combined_gdf = pd.concat([combined_gdf,gdf],ignore_index=True)

    #Save to file        
  combined_gdf = combined_gdf.to_crs('EPSG:4326')
  combined_gdf.to_file(f"{output_folder_path}/{name}.geojson", driver='GeoJSON')


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
    gdf = gdf.to_crs(epsg=3857)  # local UTM zone? This is currenlty hard coded to Utah area. This will need to be adjusted for other cities. 

    #Define area and perimeter
    gdf["area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length

    # Create feature matrix
    X = gdf[["area", "perimeter"]].values

    # Normalize scale to [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return (gdf, X_scaled)


# --- UPDATED FUNCTION ---
def k_mean_analysis(gdf, X_scaled, k_value, view, output_folder=None, filename_prefix="output"):
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init="auto")
    gdf["cluster"] = kmeans.fit_predict(X_scaled)
    """
    Runs a k-mean analysis based off the calculated k_value

    arg
    view :  1 - Plots buldinging clusters
            2 - Normalizes plot for 1
            3 - No plot.
    output_folder: folder to save output
    filename_prefix: string to use for the output filename
    """

    # ---- VIEW OPTIONS ----
    if view == 1:
        print("Plotting buildings relative to their location...")
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(column="cluster", categorical=True, legend=True, ax=ax)
        ax.set_title("K-Means Clustering: Area vs Perimeter")
        plt.show()

    elif view == 2:
        print("Plotting clusters on normalized graph...")
        plt.figure(figsize=(10,7))
        scatter = plt.scatter(
            X_scaled[:, 0],  # normalized area
            X_scaled[:, 1],  # normalized perimeter
            c=gdf["cluster"],
            cmap="tab10",
            alpha=0.7
        )
        plt.xlabel("Normalized Area")
        plt.ylabel("Normalized Perimeter")
        plt.title("K-Means Clustering of Buildings (Area vs Perimeter)")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True)
        plt.show()

    elif view == 3:
        print("No plot. Generating GeoJSON with cluster information...")

        # ---- SAVE CLUSTERED GEOJSON ----
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            # UPDATED: Uses the dynamic filename_prefix passed to the function
            output_path = os.path.join(output_folder, f"{filename_prefix}_kmeans_cluster.geojson")
        else:
            base, ext = os.path.splitext(gdf.__geo_interface__["name"] if hasattr(gdf, "__geo_interface__") else "output")
            output_path = base + "_clustered.geojson"

        gdf.to_file(output_path, driver="GeoJSON")
        print(f"Clustered GeoJSON saved to: {os.path.abspath(output_path)}")

    return gdf

def view_geojson_with_clusters(gdf, geojson_path, bb_folder=None, output_html=None, zoom_start=14):
    """
    Create a Folium HTML map with cluster polygons and optional bounding boxes.

    Args:
        gdf (GeoDataFrame): GeoDataFrame with cluster polygons
        geojson_path (str or Path): Original GeoJSON path (for naming HTML file)
        bb_folder (str or Path, optional): Folder containing bounding box .txt files
        output_html (str, optional): Output HTML file path. If None, automatically derived from geojson_path
        zoom_start (int): Initial zoom for the map
    """


    # --- Ensure CRS for centroid computation ---
    gdf_proj = gdf.to_crs(epsg=3857)
    centroid = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    center = [centroid.y.mean(), centroid.x.mean()]

    # --- Convert to WGS84 for Folium ---
    gdf = gdf.to_crs(epsg=4326)

    # --- Create Folium map ---
    m = folium.Map(location=center, zoom_start=zoom_start, tiles="cartodbpositron")

    # --- Add cluster polygons ---
    cmap = matplotlib.colormaps.get_cmap("tab10")
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

    folium.GeoJson(
        data=gdf.to_json(),
        name="Building Clusters",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=["cluster"]),
    ).add_to(m)

    # --- Add bounding boxes from txt files, if given ---
    if bb_folder is not None:
        bb_folder = Path(bb_folder)
        txt_files = list(bb_folder.glob("*.txt"))
        if txt_files:
            random.seed(42)  # consistent colors
            for txt_file in txt_files:
                with open(txt_file, "r") as f:
                    line = f.readline().strip()
                    try:
                        west, south, east, north = map(float, line.split(","))
                        # Ensure correct order
                        west, east = min(west, east), max(west, east)
                        south, north = min(south, north), max(south, north)
                    except ValueError:
                        print(f"Skipping invalid file: {txt_file}")
                        continue

                color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                folium.Rectangle(
                    bounds=[[south, west], [north, east]],
                    color=color,
                    fill=False,
                    weight=3,
                    popup=txt_file.stem
                ).add_to(m)
        else:
            print(f"No bounding box txt files found in {bb_folder}")

    folium.LayerControl().add_to(m)

    # --- Determine output HTML path ---
    if output_html is None:
        base = Path(geojson_path).stem
        output_html = str(Path(geojson_path).parent / f"{base}_clusters.html")

    m.save(output_html)
    print(f"\nMap saved to: {output_html}")
    return str(output_html)

# --- UPDATED FUNCTION ---
def run_kmeans_cluster_view(geojson_path, bb_folder=None, view=3):
    """
    This will run the whole process to create a html file of the kmeans analysis. 
    
    Args:
        geojson_path: (str): path to geojson with building footprint data 
        bb_folder: (str, optional): Adds bounding boxes given in long/lat to HTML file.
        view: View mode (1, 2, or 3)
    """
    # 1. Prepare data
    gdf, X_scaled = k_mean_prepare_area_perimeter_scaled(geojson_path)
    
    # 2. Get optimal K
    k_value = k_value_elbow_method(X_scaled, plot=False)
    
    # 3. Get proper filename from the path (e.g., "logan_utah")
    file_name_clean = os.path.splitext(os.path.basename(geojson_path))[0]
    output_folder = os.path.dirname(geojson_path)
    
    # 4. Run Analysis with dynamic name
    gdf = k_mean_analysis(
        gdf, 
        X_scaled, 
        k_value, 
        view=view, 
        output_folder=output_folder, 
        filename_prefix=file_name_clean
    )
    
    # 5. Generate Map
    html_path = view_geojson_with_clusters(gdf, geojson_path, bb_folder)
    return html_path        



#### Main Execution ####

# 1. Configuration
output_folder_path = "buildingfootprint/provo_orem_area"
target_city = "provo_central_byu_ppa" #Replace with city, or a custom name is using a custom polygon
view = 2 

# --- NEW: Custom Polygon Configuration ---
use_custom_polygon = True 

# Define your polygon vertices: [Longitude, Latitude]
# IMPORTANT: 
# 1. Order is [Lon, Lat] (x, y)
# 2. You must close the loop (Last point must equal First point)
# Use geojson.io polygon tool
custom_coords = [
    
             [
              -111.7131144153063,
              40.27453150941645
            ],
            [
              -111.71508742139163,
              40.274867311251654
            ],
            [
              -111.71735295539513,
              40.27503640265684
            ],
            [
              -111.71465451944833,
              40.271307739132084
            ],
            [
              -111.70531664848055,
              40.26162247666963
            ],
            [
              -111.69378453425075,
              40.25087925272925
            ],
            [
              -111.68657290155708,
              40.24113246621573
            ],
            [
              -111.68474908153736,
              40.24196769099623
            ],
            [
              -111.68175685821178,
              40.24297106428122
            ],
            [
              -111.67818193245014,
              40.24397368415609
            ],
            [
              -111.67438824762179,
              40.24525506995488
            ],
            [
              -111.67234574881228,
              40.24692576809261
            ],
            [
              -111.6706681727987,
              40.24915295697036
            ],
            [
              -111.6694285002074,
              40.25260443680742
            ],
            [
              -111.667021991507,
              40.25421855649196
            ],
            [
              -111.66468865917652,
              40.25622206175174
            ],
            [
              -111.66396020001342,
              40.25922671689881
            ],
            [
              -111.66746055839968,
              40.26234222409781
            ],
            [
              -111.67183546314016,
              40.26539925764885
            ],
            [
              -111.67300166333254,
              40.26795548321414
            ],
            [
              -111.67402133143304,
              40.27217424768625
            ],
            [
              -111.6793388434599,
              40.272839102200464
            ],
            [
              -111.70462652360627,
              40.273067758308116
            ],
            [
              -111.7131144153063,
              40.27453150941645
            ]
]


# Clean up filename
outputname = target_city.lower().replace(" ", "_")
geojson_file = f"{output_folder_path}/{outputname}.geojson"

# 2. Get Coordinates
try:
    if use_custom_polygon:
        # Pass the custom list directly
        area_polygon = custom_coords
    else:
        # Standard City Lookup
        area_polygon = get_city_border_coordinates("Logan, Utah")
    
    # 3. Pull Data 
    # This function accepts the list of lists format natively
    pull_building_footprints(area_polygon, output_folder_path, outputname)
    
    # 4. Run Analysis
    run_kmeans_cluster_view(geojson_file, bb_folder=None, view=view)

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()