import sys
import os

# 1. Force the local directory to the BOTTOM of the search list
sys.path.append(sys.path.pop(0))

import ee
import geemap

# 2. Check if we actually got the real library
if not hasattr(ee, 'FeatureCollection'):
    raise ImportError(f"Python is still grabbing a fake 'ee' module from: {ee.__file__}. Delete that file!")

# 3. Initialize
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

# 4. Map Provo
provo_center = [40.2444, -111.6500]
Map = geemap.Map(center=provo_center, zoom=15)

# High-res NAIP for Provo
naip = (ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(ee.Geometry.Point([-111.65, 40.24]))
        .filterDate('2021-01-01', '2023-12-31')
        .first())

# Microsoft Buildings (Utah)
buildings = ee.FeatureCollection("projects/sat-io/open-datasets/ms-lbl/utah")

Map.addLayer(naip, {'bands': ['R', 'G', 'B']}, 'NAIP High-Res')
Map.addLayer(buildings, {'color': 'red'}, 'Buildings')
Map