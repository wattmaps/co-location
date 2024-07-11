library(here)
library(tidyverse)
library(sf)

options(scipen = 999)

# Define csv file names
filepaths <- c('Cambium22Midcase_NoPhaseout_2022_Advanced_2025_120_1.csv', 'Cambium22Electrification_NoPhaseout_2022_Advanced_2025_120_2.csv', 'Cambium22Midcase_YesPhaseout_2022_Advanced_2025_120_3.csv', 'Cambium22Electrification_YesPhaseout_2022_Advanced_2025_120_4.csv','Cambium22Midcase_NoPhaseout_2022_Advanced_2025_100_5.csv', 'Cambium22Electrification_NoPhaseout_2022_Advanced_2025_100_6.csv', 'Cambium22Midcase_YesPhaseout_2022_Advanced_2025_100_7.csv', 'Cambium22Electrification_YesPhaseout_2022_Advanced_2025_100_8.csv', 'Cambium22Midcase_NoPhaseout_2022_Advanced_2030_120_9.csv', 'Cambium22Electrification_NoPhaseout_2022_Advanced_2030_120_10.csv', 'Cambium22Midcase_NoPhaseout_2022_Advanced_2030_100_11.csv', 'Cambium22Electrification_NoPhaseout_2022_Advanced_2030_100_12.csv', 'Cambium22Midcase_YesPhaseout_2022_Advanced_2030_120_13.csv', 'Cambium22Electrification_YesPhaseout_2022_Advanced_2030_120_14.csv','Cambium22Midcase_YesPhaseout_2022_Advanced_2030_100_15.csv','Cambium22Electrification_YesPhaseout_2022_Advanced_2030_100_16.csv')

# Initalize empty lists
csv_list <- list()
windOnly_csv_list <- list()
join_csv_list <- list()

# Loop through each filepath
# Read CSV and add to csv_list list
for (i in filepaths){
  csv <- read_csv(here::here('results', 'HPCscenarios', i))
  csv_list[[i]] <- csv
}

# Loop through each filepath
# Read CSV and add to windOnly_csv_list list
for (i in filepaths){
  csv <- read_csv(here::here('results', 'windOnlyScenarios', i))
  csv <- csv %>% rename_with(~ paste0('windOnly_', .), c('revenue', 'cost', 'profit', 'LCOE', 'LVOE', 'NVOE'))
  windOnly_csv_list[[i]] <- csv
}

# Initialize count
index <- seq_along(csv_list)

# Loop through each object in csv_list and windOnly_csv_list
# Add to join_csv_list list
for (i in index){
  # Extract DataFrame as tibble
  df <- as_tibble(csv_list[[i]])
  
  # Drop column
  df <- df %>% select(-c(...1))
  
  # Extract DataFrame as tibble
  windOnly_df <- as_tibble(windOnly_csv_list[[i]])
  
  # Select columns of interest
  windOnly_df <- windOnly_df %>% select(PID, windOnly_cost, windOnly_revenue, windOnly_profit, windOnly_LCOE, windOnly_LVOE, windOnly_NVOE)
  
  # Execute left join based on 'PID' column
  join_df <- left_join(df, windOnly_df, by = 'PID')
  
  # Calculate change in profit
  join_df <- join_df %>%
    mutate(diff_profit = NVOE - windOnly_NVOE,
           percent_profit = (diff_profit / windOnly_NVOE)*100) %>%
    mutate(scenario = i) %>%
    arrange(PID)
  
  join_csv_list[[i]] <- join_df
}

# Remove objects from environment
rm(df, join_df, windOnly_df, csv)

# Unlist tibbles from join_csv_list list
for (i in index){
  assign(paste0('scenario_', i), as_tibble(join_csv_list[[i]]))
}

# Concatenate all tibbles extracted join_csv_list list
all_scenarios <- bind_rows(mget(paste0('scenario_', 1:16)))

# Remove objects from environment
rm(list = paste0('scenario_', 1:16))

# Set variable to factor class
all_scenarios <- all_scenarios %>% mutate(scenario = as.factor(scenario))

# Define related scenario variables indices
index_2025_cost <- c('1', '2', '3', '4', '5', '6', '7', '8')
index_no_ptc <- c('1', '2', '5', '6', '9', '10', '11', '12')
index_midcase <- c('1', '3', '5', '7', '9', '11', '13', '15')
index_tx_availability <- c('1', '2', '3', '4', '9', '10', '13', '14')

# Assign related scenario variables
all_scenarios <- all_scenarios %>%
  mutate(capital_cost = as.factor(if_else(scenario %in% index_2025_cost, '2025', '2030')),
         ptc = as.factor(if_else(scenario %in% index_no_ptc, 'No phaseout', 'Phaseout')),
         cambium = as.factor(if_else(scenario %in% index_midcase, 'Midcase', 'Electrification')),
         tx_availability = as.factor(if_else(scenario %in% index_tx_availability, '120', '100')))

# Subset to keep sites with positive difference in profit 
subset_scenarios <- all_scenarios %>% filter(percent_profit >= 0.01)

# Read US PIDs CSV
us_pids <- read_csv(here::here('data', 'uswtdb', 'us_PID_cords_15.csv'))

us_pids_df <- us_pids %>%
  # Remove non-numeric characters except comma and minus
  mutate(geometry = gsub('[^0-9.,-]', '', geometry)) %>%
  # Split into two columns
  separate(geometry, into = c('lon', 'lat'), sep = ',') %>%
  # Convert to numeric
  mutate(across(c(lon, lat), as.numeric)) %>%
  select(-c('p_cap', 't_count', 't_cap'))

# Perform left join based on PID
subset_scenarios <- left_join(subset_scenarios, us_pids_df, by = 'PID')

# Set variables to factor class
subset_scenarios <- subset_scenarios %>%
  mutate(across(c(ptc, scenario, capital_cost, cambium, tx_availability), as.factor))

# Write as CSV
write_csv(subset_scenarios, here::here('results', 'subset_scenarios.csv'))