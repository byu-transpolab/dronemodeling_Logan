import ee
import time

# 1. Initialize Earth Engine
try:
    ee.Initialize(project="provo-test")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="provo-test")

# 2. Define the Coordinate (Provo, Utah)
point = ee.Geometry.Point([-111.6670935, 40.2432599])
region = point.buffer(100).bounds() 

# 3. Get the Best Available Imagery
collection = ee.ImageCollection('USDA/NAIP/DOQQ') \
    .filterBounds(point) \
    .sort('system:time_start', False)

if collection.size().getInfo() == 0:
    print("No NAIP imagery found for this location.")
else:
    image = collection.first()
    native_scale = image.projection().nominalScale().getInfo()
    date_str = image.date().format('YYYY-MM-dd').getInfo()

    print(f"Found image from: {date_str}")
    
    # 4. Define Export Task
    # Note: 'description' acts as the Task Name in GEE, 'fileNamePrefix' is the actual file name.
    description_str = f'Provo_Export_{date_str}'.replace('-', '_') # Safe formatting
    
    task = ee.batch.Export.image.toDrive(
        image=image.select(['R', 'G', 'B']),
        description=description_str, 
        folder='GEE_Exports',
        fileNamePrefix=f'Provo_NAIP_{date_str}', # Explicit filename
        region=region,
        scale=native_scale,
        crs='EPSG:3857',
        maxPixels=1e8
    )

    # 5. Start and MONITOR the Task
    task.start()
    print(f"Task started. ID: {task.id}")
    print("Waiting for export to complete (this takes 1-2 minutes)...")

    while task.active():
        print(f"Status: {task.status()['state']}...")
        time.sleep(5)

    # 6. Final Status Check
    status = task.status()
    if status['state'] == 'COMPLETED':
        print("\nSUCCESS! Image exported to Google Drive folder 'GEE_Exports'.")
    else:
        print("\nFAILED.")
        print(f"Error Message: {status.get('error_message', 'Unknown error')}")

'''
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
'''