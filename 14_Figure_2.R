library(here)
library(tidyverse)
library(paletteer)
library(sf)
library(stars)
library(gstat)

# Read CSV and select scenarios of interest
subset_scenarios <- readr::read_csv(here::here('results', 'subset_scenarios.csv')) %>%
  dplyr::filter(scenario %in% c(1, 2, 5, 6, 7, 9, 10, 11, 12)) %>%
  dplyr::mutate(scenario = as.factor(scenario)) %>%
  # Convert to sf object with CRS as NAD83
  sf::st_as_sf(coords = c('lon', 'lat'), crs = 4269)

# Extract world polygon from spData
north_america <- sf::st_read(system.file('shapes/world.gpkg', package = 'spData')) %>%
  # Set CRS to NAD83
  sf::st_transform(4269) %>%
  # Filter to keep polygons for Canada, Mexico, and the U.S. 
  dplyr::filter(name_long %in% c('Canada', 'Mexico', 'United States')) %>%
  # Set CRS to NAD83 (overkill)
  sf::st_transform(4269)

# Set CRS to NAD83 for U.S. state polygons
us_states <- spData::us_states %>% sf::st_transform(4269)

# Gaussian process prediction ----
# Disable spherical geometry for spatial operations
sf::sf_use_s2(FALSE)

# Create 10 x 10 km grid over contiguous U.S.
grid <- sf::st_bbox(us_states) %>%
  stars::st_as_stars(dx = 10000) %>%
  sf::st_crop(us_states)

# Find inverse distance interpolated values
i <- gstat::idw(solar_wind_ratio~1, subset_scenarios, grid)

# Proximity interpolation ----
# Create a tessellated surface
th  <- dirichlet(as.ppp(subset_scenarios))
# Error: Only projected coordinates may be converted to spatstat class objects

# Map of solar-wind ratio distribution ----
ggplot() + 
  geom_sf(data = north_america, col = 1, fill = '#e5e5e5') +
  geom_sf(data = us_states, size = 8) +
  coord_sf(xlim = c(-125, -67), ylim = c(25.75, 50)) +
  theme(panel.background = element_rect(fill = '#ADD8E6'),
        axis.text = element_blank(),
        panel.grid = element_blank(),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        plot.background = element_rect(fill = 'transparent', color = NA),
        legend.position = 'bottom', 
        legend.title = element_blank(),
        legend.background = element_rect(fill = '#F5F9FA', color = '#333533'),
        legend.text = element_text(size = 6)) +
  scale_x_continuous(breaks = c(-120, -70)) +
  scale_y_continuous(breaks = c(25, 50)) +
  labs(x = NULL, y = NULL)