library(here)
library(tidyverse)
library(paletteer)
library(ggrepel)
library(patchwork)

subset_scenarios <- read_csv(here::here('results', 'subset_scenarios.csv'))

sample <- subset_scenarios %>%
  filter(scenario %in% c(1, 2, 5, 6, 7, 9, 10, 11, 12)) %>%
  mutate(scenario = as.factor(scenario))

# Box and whisker plot for solar-wind ratio distribution ----
fig_1_1 <- sample %>%
  ggplot(aes(y = solar_wind_ratio, group = scenario, fill = scenario)) + 
  geom_boxplot(alpha = 0.7) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = 'Solar to Wind Ratio') +
  theme_classic() +
  theme(legend.position = 'top',
        legend.key.width = unit(2, 'cm'),
        legend.spacing.y = unit(5, 'cm'),
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

# Box and whisker plot for difference in co-location versus wind-only profits ----
fig_1_2 <- sample %>%
  ggplot(aes(y = diff_profit, group = scenario, fill = scenario)) + 
  geom_boxplot(alpha = 0.7) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = 'Difference in Profit (USD)') +
  theme_classic() +
  theme(legend.position = 'non',
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

# Box and whisker plot for total amount of electricity production ----
fig_1_3 <- sample %>%
  ggplot(aes(y = exportGen_lifetime, group = scenario, fill = scenario)) + 
  geom_boxplot(alpha = 0.7) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = 'Lifetime Export Generation') +
  theme_classic() +
  theme(legend.position = 'non',
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

# Stitch box and whisker plots into Figure 1 ----
fig_1_1 / fig_1_2 / fig_1_3
