library(here)
library(tidyverse)
library(sf)
library(sp)
library(gstat)
library(spatstat)
library(terra)
library(stars)

# Disable spherical geometry for spatial operations
sf::sf_use_s2(FALSE)

# Read CSV and select scenarios of interest
subset_scenarios <- readr::read_csv(here::here('results', 'all_new_scenarios.csv')) %>%
  dplyr::filter(scenario %in% c(5)) %>% # c(1, 2, 5, 6, 7, 9, 10, 11, 12)) %>%
  # dplyr::group_by(PID) %>%
  dplyr::mutate(scenario = as.factor(scenario)) %>%
  sf::st_as_sf(coords = c('lon', 'lat'), crs = 4269)

mapview::mapview(subset_scenarios['solar_wind_ratio'])

# Set CRS to NAD83 for U.S. state polygons
us_states <- spData::us_states %>% sf::st_transform(4269)
sf::st_bbox(us_states)

# Transform coordinates to projected CRS
subset_scenarios_projected <- sf::st_transform(subset_scenarios, crs = '+proj=lcc +lat_0=0 +lon_0=-100 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs') # Change the CRS to your suitable projected CRS

#### APPROACH 1 ####
# Create tessellated surface
# Minimizes edges of circumfurence to area - making it as circular as possible
th <- spatstat.geom::dirichlet(as.ppp(subset_scenarios_projected)) %>% sf::st_as_sfc() %>% sf::st_as_sf()

# Add projection information
st_crs(th) <- sf::st_crs(subset_scenarios_projected)

# Add attribute information from point data layer
# Join point attributes to polygons
th2 <- sf::st_join(th, subset_scenarios_projected, fn = mean)

# Clip tessellated  surface to geography boundaries (i.e., us_states)
us_states_projected <- sf::st_transform(us_states, crs =  st_crs(th2))
th_clip <- sf::st_intersection(th2, us_states_projected)

# Create empty grid where n is total number of cells
# Convert sf object to sp object
subset_scenarios_sp <- as(subset_scenarios_projected, 'Spatial')

# Use spsample function on sp object
grd <- as.data.frame(sp::spsample(subset_scenarios_sp, 'regular', n = 50000))

# Set names of layers
names(grd) <- c('X', 'Y')
# Set spatial coordinates
sp::coordinates(grd) <- c('X', 'Y')
# Create SpatialPixel object
sp::gridded(grd) <- TRUE
# Create SpatialGrid object
sp::fullgrid(grd) <- TRUE
# Set CRS
terra::crs(grd) <- terra::crs(subset_scenarios_sp)

# Point interpolation
P_idw <- gstat::idw(solar_wind_ratio ~ 1, subset_scenarios_sp, newdata = grd, idp = 2.0)

# Create a SpatRaster
r <- terra::rast(P_idw)
# Mask values in a SpatRaster
r_m <- terra::mask(r, st_as_sf(us_states_projected))


#### APPROACH 2 ####
# Create 10 x 10 km grid over contiguous U.S.
grid <- sf::st_bbox(us_states_projected) %>%
  stars::st_as_stars(dx = 10000) %>%
  sf::st_crop(us_states_projected)

i <- gstat::idw(solar_wind_ratio ~ 1, subset_scenarios_sp, grid)
ggplot() + geom_stars(data = i)

# Calculate sample variogram
v <- gstat::variogram(solar_wind_ratio ~ 1, subset_scenarios_sp)
v_m <- gstat::fit.variogram(v, gstat::vgm(psill = 1, model = 'Exp', range = 10000, kappa = 1))
k <- gstat::krige(solar_wind_ratio ~ 1, subset_scenarios_sp, grid, v_m)
ggplot() + geom_stars(data = k, aes(fill = var1.pred, x = x, y = y))
