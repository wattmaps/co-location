library(here)
library(tidyverse)
library(sf)
library(tmap)
library(tmaptools)

# Define filepath
uswtdb_dir <- here::here('data', 'uswtdb')

# Read USWTDB
wind_proj <- read_csv(paste0(uswtdb_dir, '/uswtdb_v4_3_20220114_ucsb.csv'))

# Add spatial coordinates
wind_proj_geom <- sf::st_as_sf(wind_proj, coords = c('xlong', 'ylat'), crs = 4269)

# class(wind_proj_geom$geometry)

# Create bounding box for contiguous U.S.
contiguous_us <- sf::st_bbox(c(xmin = -125, ymin = 24, xmax = -66.5, ymax = 50), crs = st_crs(wind_proj_geom))
wind_proj_contiguous <- sf::st_crop(wind_proj_geom, contiguous_us)

# View 
# tm_shape(wind_proj_contiguous) +
#  tm_dots(col = '#2A9D8F') +
#  tm_layout(main.title = 'Exisiting Wind Sites in the Contiguous U.S.')

# Save as shapefile
# st_write(wind_proj_contiguous, here::here('data', 'existing_wind_contiguous_us.shp'))

# Group by project ID 
pid_data <- wind_proj_contiguous %>% 
  group_by(p_id) %>%
  # Keep project capacity
  summarise(p_cap = first(p_cap)) %>%
  # Create polygon from point data
  mutate(polygon = st_convex_hull(geometry)) %>%
  # Find centroid of polygon
  mutate(centroid = st_centroid(polygon)) %>%
  filter(!is.na(p_id))

# Set view to interactive
# tmap_mode(mode = 'view')

# Plot to check that centroids/polygons are accurate
# tm_shape(pid_data$geometry) +
#  tm_dots(col = '#264653', size = 0.2) +
#  tm_shape(pid_data$polygon) +
#  tm_fill(col = '#2a9d8f', alpha = 0.5) +
#  tm_shape(pid_data$centroid) +
#  tm_dots(col = '#e76f51')

# Select variables needed for SAM simulation
pid_site_coords <- pid_data %>%
  select(c(p_cap, centroid)) %>% 
  mutate(centroid = st_transform(centroid, crs = 4269))

# Subset and drop spatial class
pid_coords <- pid_site_coords %>%
  mutate(PID = row_number()) %>%
  mutate(lon = unlist(map(pid_site_coords$centroid, 1)),
         lat = unlist(map(pid_site_coords$centroid, 2))) %>% 
  select(c(PID, p_cap, lon, lat)) %>% 
  sf::st_drop_geometry()

# Save df to csv
write_csv(pid_coords, paste0(uswtdb_dir, '/us_PID_cords.csv'), row.names = FALSE)