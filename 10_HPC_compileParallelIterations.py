''' ============================
Import packages and set directory
============================ '''

import os, sys
import glob
import time
import pandas as pd
import numpy as np
import shutil

start_time = time.time()

current_dir = os.getcwd()
print(current_dir)

inputFolder = os.path.join(current_dir, 'data')

seq = pd.read_csv(os.path.join(inputFolder, 'uswtdb', 'us_PID_cords_15.csv'))

# Create a df with column names
output_df = pd.DataFrame(columns = ['PID', 'solar_capacity', 'wind_capacity', 'solar_wind_ratio', 'tx_capacity', 'batteryCap', 'batteryEnergy', 'revenue', 'cost', 'profit'])

''' ============================
Retrieve system arguments
============================ '''

cambium_scen = sys.argv[1] ## should be either 'Cambium22Electrification' or "Cambium22Midcase"
PTC_scen = sys.argv[2] ## should be either "NoPhaseout" or "YesPhaseout"
ATBreleaseYr_scen = sys.argv[3] ## should be either 2022 or 2023
ATBcost_scen = sys.argv[4] ## should be advanced or moderate
ATBcapexYr_scen = sys.argv[5] ## should be either "2025" or "2030"
tx_scen = sys.argv[6] ## should be either "100" or "120"
scen_num = sys.argv[7]
mode = sys.argv[8] ## anything--can be "check" or "final"

''' ============================
Set filepath for combined results
============================ '''

scenario_foldername_iter = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num
scenario_filename_combined = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num  + ".csv"
scenario_filename_combined_missingPIDs = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num + "_"  + mode + "_missingPIDs" + ".csv"

output_df_path_iterationsFolder = os.path.join(current_dir, 'results', 'HPCscenarios', scenario_foldername_iter)
output_df_path = os.path.join(current_dir, 'results', 'HPCscenarios', scenario_filename_combined)
output_df_path_missingPIDs = os.path.join(current_dir, 'results', 'HPCscenarios', scenario_filename_combined_missingPIDs)

''' ============================
Combine into singular DataFrame
============================ '''
    
print("Concatenating iterations into one csv and save to drive")
all_csvs_iter = glob.glob(os.path.join(output_df_path_iterationsFolder, "*.csv"))
combined_df = pd.concat((pd.read_csv(f) for f in all_csvs_iter), axis = 0, ignore_index=True)
combined_df.to_csv(output_df_path)

# Produce list of PIDs that are still missing/yet to be run
missingPIDs_list = list(set(seq['PID'].tolist()) - set(combined_df['PID'].tolist()))

print("Missing PIDs:", missingPIDs_list)
missingPIDs_df = pd.DataFrame(missingPIDs_list, columns=['missingPIDs'])
missingPIDs_df.to_csv(output_df_path_missingPIDs)

end_time = time.time()

print('Time taken:', end_time - start_time, 'seconds')