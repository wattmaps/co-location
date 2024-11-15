library(here)
library(tidyverse)
library(sf)
library(patchwork)

options(scipen = 999)

# Read CSV and select scenarios of interest
subset_scenarios <- readr::read_csv(here::here("results", "all_new_scenarios.csv")) %>%
  dplyr::mutate(scenario = as.factor(scenario)) %>%
  sf::st_as_sf(coords = c("lon", "lat"), crs = 4269) %>% 
  filter(scenario %in% c(1, 3, 5, 10))

us_states <- spData::us_states %>% 
  sf::st_transform(4269) %>%
  dplyr::select(-c(GEOID, REGION, AREA, total_pop_10, total_pop_15)) %>%
  rename(state = NAME)

subset_scenarios <- st_intersection(subset_scenarios, us_states)

scenarios_by_state <- subset_scenarios %>%
  group_by(state, scenario) %>%
  summarise(mean_wind_capacity = mean(wind_capacity, na.rm = TRUE),
            mean_potentialGen_solar = mean(potentialGen_solar_lifetime, na.rm = TRUE),
            mean_batteryCap = mean(batteryCap, na.rm = TRUE)) %>%
  filter(scenario %in% c(1, 3, 5, 10)) %>%
  st_drop_geometry()

scenarios_by_state <- left_join(us_states, scenarios_by_state, by = "state") %>%
  drop_na(mean_wind_capacity)
  
pal <- c("#FAE9A0FF", "#DBD797FF", "#BCC68DFF", "#9CB484FF", "#7DA37BFF", "#5E9171FF", "#3F7F68FF", "#1F6E5EFF", "#005C55FF")

row_1 <- ggplot() +
  geom_sf(data = us_states, size = 8) +
  geom_sf(data = subset_scenarios, aes(color = solar_wind_ratio)) +
  facet_wrap(~scenario, ncol = 4) +
  scale_color_gradientn(colors = pal, name = "Solar to Wind Ratio",
                        guide = guide_colorbar(label.position = "bottom",
                                               title.position = "top",
                                               direction = "horizontal",
                                               barwidth = unit(6, "cm"),
                                               barheight = unit(0.5, "cm"))) +
  theme_void() +
  theme(legend.position = "top") +
  labs(x = NULL, y = NULL)

row_2 <- ggplot() + 
  geom_sf(data = us_states, fill = "#353535", size = 8) +
  geom_sf(data = scenarios_by_state, aes(fill = mean_wind_capacity)) +
  facet_wrap(~scenario, ncol = 4) +
  # scale_fill_paletteer_c(palette = "Redmonder::sPBIYlGn") +
  scale_fill_gradientn(colors = pal, name = "Mean Wind Capacity ()", 
                       guide = guide_colorbar(label.position = "bottom",
                                              title.position = "top",
                                              direction = "horizontal",
                                              barwidth = unit(6, "cm"),
                                              barheight = unit(0.5, "cm"))) +
  theme_void() +
  theme(legend.position = "top") +
  labs(x = NULL, y = NULL)

row_3 <- ggplot() + 
  geom_sf(data = us_states, fill = "#353535", size = 8) +
  geom_sf(data = scenarios_by_state, aes(fill = mean_potentialGen_solar)) +
  facet_wrap(~scenario, ncol = 4) +
  scale_fill_gradientn(colors = pal, name = "Mean Potential Solar Generation ()",
                       guide = guide_colorbar(label.position = "bottom",
                                              title.position = "top",
                                              direction = "horizontal",
                                              barwidth = unit(6, "cm"),
                                              barheight = unit(0.5, "cm"))) +
  theme_void() +
  theme(legend.position = "top") +
  labs(x = NULL, y = NULL)

row_4 <- ggplot() + 
  geom_sf(data = us_states, fill = "#353535", size = 8) +
  geom_sf(data = scenarios_by_state, aes(fill = mean_batteryCap)) +
  facet_wrap(~scenario, ncol = 4) +
  scale_fill_gradientn(colors = pal, name = "Mean Battery Capacity ()",
                       guide = guide_colorbar(label.position = "bottom",
                                              title.position = "top",
                                              direction = "horizontal",
                                              barwidth = unit(6, "cm"),
                                              barheight = unit(0.5, "cm"))) +
  theme_void() +
  theme(legend.position = "top") +
  labs(x = NULL, y = NULL)

row_1 / row_2 / row_3 / row_4
