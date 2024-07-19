library(here)
library(janitor)
library(tidyverse)
library(sf)
library(nngeo)

# Read CSV file with point coordinates
us_pids <- read_csv(here::here('data','us_PID_cords_PID1335.csv'))
us_pids <- as.data.frame(us_pids) 

# Convert to shapefile DataFrame with CRS as NAD83
us_pids_geom <- st_as_sf(us_pids, coords = c('lon', 'lat'), crs = 4269)

# Associate NREL GEA region to PID ----
# Read ReEDS mapping data
ReEDS <- read_csv(here::here('data','ReEDS','ReEDS_mapping.csv')) %>%
  rename(pca = r) %>% 
  mutate(pca = as.factor(pca))

# Read ReEDS vector shapefile
ReEDS_map <- sf::st_read(here::here('data','ReEDS','reeds_shapefiles_08032021', 'US_Canada_Map.shp'))

# Create sf DataFrame
gea_df <- ReEDS_map %>% 
  # Clean column names
  janitor::clean_names() %>%
  # Convert 'pca' class to factor
  mutate(pca = as.factor(pca)) %>%
  # Join with ReEDS data
  # Keep only PCA regions in GEA regions
  right_join(ReEDS, by = 'pca') %>%
  # Convert 'gea' class to factor
  mutate(gea = as.factor(gea)) %>%
  # Transform CRS to NAD83
  sf::st_transform(4269)

# Disable spherical geometry for spatial operations
sf_use_s2(FALSE)

# Find GEA regions of point coordinates, create sf DataFrame
point_cords_gea <- sf::st_join(us_pids_geom, gea_df, join = st_within)

# Select columns of interest in sf DataFrame
pids_gea_df <- point_cords_gea %>%
  dplyr::select(PID, t_cap, t_count, t_rd, pca, gea, geometry)

# Check NA values
# pids_gea_df[is.na(pids_gea_df$pca), ]

# Assign GEA regions to specific rows with NA values
pids_gea_df[39, 6] <- 'NEWEc'
pids_gea_df[151, 6] <- 'RFCEc'
pids_gea_df[228, 6] <- 'NYSTc'
pids_gea_df[787, 6] <- 'NYSTc'
pids_gea_df[827, 6] <- 'ERCTc'

# Extract longitude and latitude from geometry and drop geometry column
pids_gea_df <- pids_gea_df %>%
  mutate(lon = unlist(map(pids_gea_df$geometry, 1)),
         lat = unlist(map(pids_gea_df$geometry, 2))) %>%
  sf::st_drop_geometry()

# Calculate potential installed capacity of PID ----
# Filter and calculate various capacities 
PID_updated <- us_pids %>% 
  filter(!is.na(t_cap)) %>% 
  mutate(
    # Calculate project wind capacity in kW
    p_cap_kw = t_cap*t_count,
    # Calculate project wind capacity in mW
    p_cap_mw = p_cap_kw/1000,
    # Calculate project solar capacity in mW
    solar_installed_cap_mw = round(p_cap_mw*(50/3), 3))

# Find mean project capacity in MW
# Note: mean project capacity in MW used later for projects without system size
PID_updated_summary <- PID_updated %>% 
  summarize(avg_p_cap_mw = mean(p_cap_mw))

# Store mean p_cap_mw as single value
avg_p_cap <- as.numeric(PID_updated_summary[1, 'avg_p_cap_mw'])

# Find PIDs without system size, substitute mean size 
PID_na <- PID %>% 
  filter(is.na(t_cap)) %>% 
  mutate(p_cap_kw = avg_p_cap*1000,
         p_cap_mw = avg_p_cap,
         solar_installed_cap_mw = round(avg_p_cap*(50/3),3))

# Concatenate to DataFrame
PID_clean <- rbind(PID_updated, PID_na) 

# Order by PID
PID_clean <- PID_clean[order(PID_clean$PID),]

# Subset solar installed capacity
solar_cf_by_PID <- PID_clean %>% 
  select(c(PID, solar_installed_cap_mw))
# Write csv of solar installed capacity
write_csv(solar_cf_by_PID, here::here('data', 'Potential_Installed_Capacity', 'solar_land_capacity.csv')) 

# Subset wind installed capacity
wind_cf_by_PID <- PID_clean %>% 
  select(c(PID, p_cap_mw))
# Write csv of wind installed capacity
write_csv(wind_cf_by_PID, here::here('data', 'Potential_Installed_Capacity', 'wind_land_capacity.csv'))

# Associate transmission capacity  ----
# Read substation data and filter based on minimum voltage
substation <- sf::st_read(here::here('data', 'HIFLD', 'HIFLD2020.gdb')) %>%
  # Set CRS of object to NAD83
  sf::st_transform(4269) %>% 
  # Filter for voltage greater than or equal to 115
  filter(MIN_VOLT >= 115) %>% 
  janitor::clean_names() 

# Find substation observation that is nearest neighbor of each project
substation_join <- st_join(us_pids_geom, substation, join = st_nn, k = 1, progress = FALSE) %>%
  # Keep only columns of interest
  dplyr::select(PID, id, latitude, longitude, lines, max_volt, min_volt, geometry) %>%
  # Rename variables to common keys
  rename(SID = id, 
         SID_lat = latitude, 
         SID_lon = longitude,
         PID_geom = geometry) %>%
  # Organize in ascending order
  arrange(PID)

# Extract nearest neighbor substation observations (SIDs)
sids <- substation_join %>% 
  # Remove vector geometry 
  st_drop_geometry() %>%
  # Keep only relational keys
  dplyr::select(PID, SID)

# Extract geometries for nearest neighbor substation observations (SIDs)
sids_shapes <- substation %>%
  # Keep only relational key and associated geometry 
  dplyr::select(id, Shape) %>% rename(SID = id) %>%
  # Join to nearest neighbor substation observations
  inner_join(sids, by = 'SID') %>%
  # Keep only columns of interest
  dplyr::select(PID, SID, Shape) %>%
  # Organize in ascending order
  arrange(PID)

substation_pids <- substation_join %>%
  # Bind geometry column to initial join
  bind_cols(sids_shapes) %>%
  rename(SID_geom = Shape,
         PID = PID...1, 
         SID = SID...2) %>% 
  select(-c(PID...9, SID...10, SID_lat, SID_lon))

substation_pids_dist <- substation_pids %>%
  # Compute distance from project to substation geometry in meter and kilometer
  mutate(distance_m = st_distance(PID_geom, SID_geom, by_element = T),
         distance_km = units::set_units(distance_m, 'km')) %>%
  # Remove units
  units::drop_units()

# Extract mean
mean(substation_pids_dist$distance_km)

# Drop geometry from sf DataFrame
substation_pids_df <- substation_pids_dist %>%
  sf::st_drop_geometry() %>%
  dplyr::select(-c(SID_geom))

# Add column
substation_mw_df <- substation_pids_df %>%
  mutate(substation_mw = NA) 

substation_mw_df$substation_mw <- ifelse(
  substation_mw_df$min_volt <= 138, 
  64000/1000, 
  substation_mw_df$substation_mw)

substation_mw_df <- substation_pids_df %>%
  mutate(substation_mw = case_when(
    min_volt <= 138 ~ (48000/1000),
    min_volt > 138 & min_volt <= 230 ~ (64000/1000),
    min_volt > 230 & min_volt <= 345 ~ (132000/1000),
    min_volt > 345 & min_volt <= 500 ~ (390000/1000),
    min_volt > 500 & min_volt <= 765 ~ (910000/1000)
  ))

# Write to CSV
write_csv(substation_mw_df, here::here('data', 'PID_Attributes', 'substation.csv'), row.names = FALSE)

# Filter projects by minimum capacities
us_pids_10 <- read_csv(here::here('uswtdb', 'us_PID_cords.csv')) %>% filter(p_cap >= 10)

us_pids_15 <- read_csv(here::here('uswtdb', 'us_PID_cords.csv')) %>% filter(p_cap >= 15)

us_pids_20 <- read_csv(here::here('uswtdb', 'us_PID_cords.csv')) %>% filter(p_cap >= 20)