''' ============================
Import packages and set directory
============================ '''
import os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import Polygon
from shapely.geometry import Point

thisDir = os.path.abspath(os.curdir)
print('Working directory:  ', thisDir)

''' ============================
Read and crop USWTDB point geometries
============================ '''

wind_proj = pd.read_csv(os.path.join(thisDir, 'data', 'uswtdb', 'uswtdb_v4_3_20220114_ucsb.csv'))

wind_proj_geom = gpd.GeoDataFrame(wind_proj, 
                                  geometry = gpd.points_from_xy(wind_proj['xlong'], wind_proj['ylat']),
                                  crs = 'EPSG:4269')

# Create bounding box for contiguous U.S.
bbox = box(-125, 24, -66.5, 50)

# Crop geometries for the contiguous US
wind_proj_contiguous = gpd.clip(wind_proj_geom, mask = bbox)

# Write GeoDataFrame to a shapefile
# output_shapefile_path = os.path.join(thisDir, 'data', 'existing_wind_contiguous_us.shp')
# wind_proj_contiguous.to_file(output_shapefile_path)

''' ============================
Find USWTDB project centroids
============================ '''

# Group by p_id and keep p_cap for SAM simulation and optimization
wind_proj = wind_proj_contiguous[['p_id', 'p_cap', 'geometry']]
wind_proj = wind_proj.dissolve(by = 'p_id', aggfunc = 'first').reset_index()

# Rename column
wind_proj = wind_proj.rename(columns = {'0': 'geometry'})  

# Create polygons from point geometries and find wind project centroids
centroid = wind_proj.dissolve(by = 'p_id').convex_hull.centroid.reset_index().dropna() 

# Join centroids to variable of interest (p_cap)
p_cap = wind_proj[['p_id', 'p_cap']]
pid_site_coords = pd.merge(p_cap, centroid, on = 'p_id')

# Rename column
pid_site_coords = pid_site_coords.rename(columns = {'0': 'centroid'})  

# Reset index
pid_coords = pid_site_coords.reset_index()

# Rename column
pid_coords.columns.values[3] = 'geometry'

# Set type as GeoDataFrame
pid_coords_geom = gpd.GeoDataFrame(pid_coords,geometry = pid_coords.columns[3], crs = 'EPSG:4269')

# Extract longtiude from centroid
pid_coords_geom['lon'] = pid_coords_geom.geometry.x

# Extract latitude from centroid
pid_coords_geom['lat'] = pid_coords_geom.geometry.y

# Subset and save df to csv
pid_coords_geom = pid_coords_geom[['index', 'p_cap', 'lon', 'lat']].rename(columns = {'index': 'PID'})
file_path = os.path.join(thisDir, 'data', 'uswtdb', 'us_PID_cords.csv')
pid_coords.to_csv(file_path, index = False)

''' ============================
Associate NREL GEA region to PID
============================ '''
