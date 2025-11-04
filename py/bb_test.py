from pathlib import Path
import pyproj
import folium

# Folder containing the bounding box .txt files
bb_folder = Path("/Users/willicon/Desktop/seed80_2025_11_04_081855/lat_log_bb")

# Create a projection transformer: Web Mercator -> WGS84
transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Create a base Folium map (center somewhere in Logan, Utah)
m = folium.Map(location=[41.74, -111.82], zoom_start=13)

for txt_file in bb_folder.glob("*.txt"):
    with open(txt_file, "r") as f:
        line = f.readline().strip()
        try:
            west, south, east, north = map(float, line.split(","))
        except ValueError:
            print(f"Skipping invalid file: {txt_file}")
            continue

    # Convert from EPSG:3857 -> EPSG:4326
    west_lon, south_lat = transformer.transform(west, south)
    east_lon, north_lat = transformer.transform(east, north)

    folium.Rectangle(
        bounds=[[south_lat, west_lon], [north_lat, east_lon]],
        color="red",
        fill=False,
        weight=2,
        popup=txt_file.stem
    ).add_to(m)

# Save map
m.save("log_bbox_map.html")
print("Map saved: log_bbox_map.html")
