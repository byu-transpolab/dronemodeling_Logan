import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import random
import folium

def plot_all_bboxes(bb_folder, zoom=19, basemap=True, color_seed=42):
    """
    Plot all bounding boxes from a folder on a single map.
    
    Args:
        bb_folder (str or Path): Folder containing .txt bounding box files
        zoom (int): Zoom level for basemap
        basemap (bool): Whether to overlay a basemap
        color_seed (int): Seed for random colors
    """
    bb_folder = Path(bb_folder)
    if not bb_folder.exists() or not bb_folder.is_dir():
        raise ValueError(f"{bb_folder} is not a valid folder.")
    
    txt_files = list(bb_folder.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {bb_folder}")

    # Prepare random colors
    random.seed(color_seed)
    colors = {}

    geoms = []
    labels = []

    for txt_file in txt_files:
        # Parse bounding box
        with open(txt_file, "r") as f:
            line = f.readline().strip()
            try:
                west, south, east, north = map(float, line.split(","))
            except:
                print(f"Skipping invalid file: {txt_file}")
                continue

        geoms.append(box(west, south, east, north))
        labels.append(txt_file.stem)

        # Assign a random color for each file (or cluster if embedded)
        colors[txt_file.stem] = f"#{random.randint(0, 0xFFFFFF):06x}"

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({"label": labels, "color": [colors[l] for l in labels]}, geometry=geoms, crs="EPSG:4326")

    # Plot
    ax = gdf.to_crs(epsg=3857).plot(
        edgecolor=gdf["color"],
        facecolor='none',
        linewidth=2,
        figsize=(12,12)
    )

    # Optionally add basemap
    if basemap:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=zoom)

    ax.set_title(f"All bounding boxes in {bb_folder.name}")
    plt.show()



def plot_all_bboxes_html(bb_folder, html_file):
    """
    Plot all bounding boxes from a folder on an interactive HTML map using Folium.

    Args:
        bb_folder (str or Path): Folder containing .txt bounding box files
        html_file (str or Path): Output HTML file path
    """
    bb_folder = Path(bb_folder)
    if not bb_folder.exists() or not bb_folder.is_dir():
        raise ValueError(f"{bb_folder} is not a valid folder.")
    
    txt_files = list(bb_folder.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {bb_folder}")

    # Initialize lists for coordinates and labels
    lat_list, lon_list = [], []
    boxes = []

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            line = f.readline().strip()
            try:
                west, south, east, north = map(float, line.split(","))
            except ValueError:
                print(f"Skipping invalid file: {txt_file}")
                continue
        lat_list.extend([south, north])
        lon_list.extend([west, east])
        boxes.append((west, south, east, north, txt_file.stem))

    if not lat_list or not lon_list:
        raise ValueError("No valid bounding boxes found.")

    # Map centered at mean lat/lon
    center_lat = sum(lat_list) / len(lat_list)
    center_lon = sum(lon_list) / len(lon_list)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles=None)

    # Add Esri World Imagery as basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and GIS User Community",
        name="Esri World Imagery",
        overlay=False,
        control=True
    ).add_to(m)

    # Add each bounding box with a random color
    random.seed(42)
    for west, south, east, north, label in boxes:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color=color,
            fill=False,
            weight=3,
            popup=label
        ).add_to(m)

    # Save to HTML
    m.save(html_file)
    print(f"Saved interactive map to {html_file}")

html_file = 'buildingfootprint/logan_clusters.html'
bb_folder = "/Users/willicon/Desktop/seed80_2025_11_03_145623/lat_log_bb"
plot_all_bboxes_html(bb_folder, html_file)