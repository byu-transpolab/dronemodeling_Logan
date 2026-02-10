import ee
import geemap
from IPython.display import display

# 1. Initialize
try:
    ee.Initialize(project='provo-test')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='provo-test')

# 2. Define Provo Area (Center and AOI)
provo_center = [40.2444, -111.6500]
provo_aoi = ee.Geometry.Point([-111.65, 40.24]).buffer(5000).bounds()

# 3. Load High-Res NAIP Imagery (60cm resolution)
# Filtering for the most recent flight available for Utah
naip = (ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(provo_aoi)
        .filterDate('2021-01-01', '2024-12-31')
        .sort('system:time_start', False)
        .first())

# 4. Load Building Footprints 
# Updated Path: Using the highly stable VIDA Combined USA collection
buildings = ee.FeatureCollection("projects/sat-io/open-datasets/VIDA_COMBINED/USA") \
              .filterBounds(provo_aoi)

# 5. Create Map
Map = geemap.Map(center=provo_center, zoom=15)

# Add NAIP Imagery
Map.addLayer(naip, {'bands': ['R', 'G', 'B']}, 'NAIP High-Res (Provo)')

# Add Building Footprints with specific styling
building_viz = {'color': 'red', 'fillColor': '00000000', 'width': 1.5}
Map.addLayer(buildings.style(**building_viz), {}, 'Building Footprints')

display(Map)