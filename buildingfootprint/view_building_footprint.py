from scripts import pull_building_footprint, view_geojson, Spatially_constrained_clustering, view_cluster



#Define 
output_file = "buildingfootprint/logan_building_footprints.geojson"


#Define Area of intrerest 
# Geometry copied from https://geojson.io ----Current Coordinates are for Logan, Utah
area_coordinates = [
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



#pull_building_footprint(area_coordinates,output_file)
#view_geojson(output_file)

geojson_file = "buildingfootprint/logan_building_footprints.geojson"

Spatially_constrained_clustering(geojson_file)
#view_cluster()