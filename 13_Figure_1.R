library(here)
library(tidyverse)
library(paletteer)
library(ggrepel)
library(patchwork)

# options(scipen = 10000)

subset_scenarios <- read_csv(here::here('results', 'subset_scenarios.csv'))

# Subset Midcase/BAU scenarios, PTC does not phaseout
no_ptc_scenarios <- subset_scenarios %>%
  filter(ptc == 'No phaseout') %>%
  filter(cambium == 'Midcase') %>%
  mutate(scenario = as.factor(scenario),
         tx_availability = as.factor(tx_availability),
         # Find battery capacity as a percentage of solar capacity
         pct_batteryCap = (batteryCap/solar_capacity) * 100,
         # Find annual average generation exported
         mean_annual_exportGen = exportGen_lifetime/30) 

# Subset Midcase/BAU scenarios, PTC phaseout
ptc_scenarios <- subset_scenarios %>%
  filter(ptc == 'Phaseout') %>%
  filter(cambium == 'Midcase') %>%
  mutate(scenario = as.factor(scenario),
         tx_availability = as.factor(tx_availability),
         # Find battery capacity as a percentage of solar capacity
         pct_batteryCap = (batteryCap/solar_capacity) * 100,
         # Find annual average generation exported
         mean_annual_exportGen = exportGen_lifetime/30)

# View scenario numbers 
# no_ptc_scenarios <- as.character(unique(no_ptc_scenarios$scenario))
# ptc_scenarios <- as.character(unique(ptc_scenarios$scenario))

# Box and whisker plot for solar-wind ratio distribution ----
fig_1_1 <- ptc_scenarios %>%
  ggplot(aes(y = solar_wind_ratio, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ylim(c(-0.15, 2.0)) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Ratio of solar to wind installed capacity', width = 12)) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

fig_1_2 <- no_ptc_scenarios %>%
  ggplot(aes(y = solar_wind_ratio, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ylim(c(-0.15, 2.0)) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Ratio of solar to wind installed capacity', width = 12)) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank())

# Box and whisker plot for percent profit in co-location versus wind-only profits ----
fig_1_3 <- ptc_scenarios %>%
  ggplot(aes(y = percent_profit, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ylim(c(-75, 75)) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Difference in lifetime profit (USD/MWh)', width = 12)) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

fig_1_4 <- no_ptc_scenarios %>%
  ggplot(aes(y = percent_profit, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ylim(c(-75, 75)) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Difference in lifetime profit (USD/MWh)', width = 12)) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank())

# Box and whisker plot for total amount of electricity production ----
fig_1_5 <- ptc_scenarios %>%
  ggplot(aes(y = mean_annual_exportGen, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Annual average generation exported (MWh/yr)', width = 12)) +
  scale_y_continuous(limits = c(0.00, 2000000.0), labels = scales::comma) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

fig_1_6 <- no_ptc_scenarios %>%
  ggplot(aes(y = mean_annual_exportGen, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Annual average generation exported (MWh/yr)', width = 12)) +
  scale_y_continuous(limits = c(0.00, 2000000.0), labels = scales::comma) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank())

# Box and whisker plot for battery capacity as a percentage of solar capacity ----
fig_1_7 <- ptc_scenarios %>%
  ggplot(aes(y = pct_batteryCap, x = scenario, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ylim(c(-15, 75)) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Battery capacity as percent of solar capacity', width = 12)) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank())

fig_1_8 <- no_ptc_scenarios %>%
  ggplot(aes(y = pct_batteryCap, x = scenario, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ylim(c(-15, 75)) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Battery capacity as percent of solar capacity', width = 12)) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank())

# Box and whisker plot for curtailment ----
fig_1_9 <- ptc_scenarios %>%
  ggplot(aes(y = curtailment_c, x = scenario, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Curtailment', width = 12), x = 'Scenario') +
  scale_y_continuous(limits = c(0.00, 0.35), labels = scales::comma) +
  theme_classic() +
  theme(legend.position = 'non',
        legend.title = element_blank())

fig_1_10 <- no_ptc_scenarios %>%
  ggplot(aes(y = curtailment_c, x = scenario, group = scenario, fill = tx_availability)) + 
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  paletteer::scale_fill_paletteer_d('NatParksPalettes::Olympic') +
  labs(y = str_wrap('Curtailment', width = 12), x = 'Scenario') +
  scale_y_continuous(limits = c(0.00, 0.35), labels = scales::comma) +
  theme_classic() +
  theme(legend.position = 'bottom',
        legend.title = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y = element_blank()) 

# Stitch box and whisker plots into Figure 1 ----
row_1 <- fig_1_1 + fig_1_2
row_2 <- fig_1_3 + fig_1_4
row_3 <- fig_1_5 + fig_1_6
row_4 <- fig_1_7 + fig_1_8
row_5 <- fig_1_9 + fig_1_10

row_1 / row_2 / row_3 / row_4 / row_5
