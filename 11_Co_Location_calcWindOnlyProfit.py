''' ============================
Import packages and set directory
============================ '''

# install mpi4py available on conda
# check cnsi website and check for mpi workshops

import os, sys
import glob
import time
import pandas as pd
import numpy as np
import shutil
from pyomo.environ import *
from pyomo.opt import SolverFactory

start_time = time.time()

current_dir = os.getcwd()
print(current_dir)

inputFolder = os.path.join(current_dir, 'data')

## Use specific PIDS
#seq = open(os.path.join(inputFolder, 'uswtdb', 'pids_20_MW.txt'))
seq = pd.read_csv(os.path.join(inputFolder, 'uswtdb', 'us_PID_cords_15.csv'))
PIDsList_needToRun = seq['PID'].tolist()

''' ============================
Retrieve system arguments
============================ '''

cambium_scen = sys.argv[1] ## should be either 'Cambium22Electrification' or "Cambium22Midcase"
PTC_scen = sys.argv[2] ## should be either "NoPhaseout" or "YesPhaseout"
ATBreleaseYr_scen = str(sys.argv[3]) ## should be either 2022 or 2023
ATBcost_scen = sys.argv[4] ## should be advanced or moderate
ATBcapexYr_scen = str(sys.argv[5]) ## should be either "2025" or "2030"
tx_scen = str(sys.argv[6]) ## should be either "100" or "120"
scen_num = str(sys.argv[7])
mode = sys.argv[8] ## should be either "initial" or "backfill"
# backfillNum = sys.argv[9]

# cambium_scen = "Cambium22Midcase" ## should be either 'Cambium22Electrification' or "Cambium22Midcase"
# PTC_scen = "NoPhaseout" ## should be either "NoPhaseout" or "YesPhaseout"
# ATBreleaseYr_scen = "2022" ## should be either 2022 or 2023
# ATBcost_scen = "advanced" ## should be advanced or moderate
# ATBcapexYr_scen = "2025" ## should be either "2025" or "2030"
# tx_scen = "120" ## should be either "100" or "120"
# scen_num = "1"
# mode = "initial" ## should be either "initial" or "backfill"
    
''' ============================
Initialize df and filepaths
============================ '''

# Create a df with column names
# Create a df with column names
output_df = pd.DataFrame(columns = ['PID',\
                                    'solar_capacity', \
                                    'wind_capacity', \
                                    'solar_wind_ratio', \
                                    'tx_capacity', \
                                    'batteryCap', \
                                    'batteryEnergy', \
                                    'revenue', \
                                    'cost', \
                                    'profit',\
                                    'LCOE',\
                                    'LVOE',\
                                    'NVOE',\
                                    'potentialGen_wind_lifetime',\
                                    'potentialGen_solar_lifetime',\
                                    'solar_wind_ratio_potentialGen', \
                                    "actualGen_lifetime",\
                                    "potentialGen_lifetime",\
                                    "export_lifetime",\
                                    "curtailment"])

scenario_foldername = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num
scenario_filename_combined = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num  + ".csv"

output_df_path_csv = os.path.join(current_dir, 'results', 'windOnlyScenarios', scenario_filename_combined)

''' ============================
reassign variables 
============================ '''

## Reassign variables to match 
if cambium_scen == "Cambium22Electrification":
    cambium_scen = "Cambium22_Electrification"
if cambium_scen == "Cambium22Midcase":
    cambium_scen = "Cambium22_Mid-case"

if PTC_scen == "NoPhaseout":
    PTC_scen = "Cash_Flow_PTC_No_Phaseout"
if PTC_scen == "YesPhaseout":
    PTC_scen = "Cash_Flow"
    
if tx_scen == "100":
    tx_scen = 1.0
if tx_scen == "120":
    tx_scen = 1.2
    
if ATBcapexYr_scen == "2030":
    cambium_scen_yr_append = "_2030"
if ATBcapexYr_scen == "2025":
    cambium_scen_yr_append = ""
    
''' ============================
Read data
============================ '''

# Set file path and read csv for substation attributes
pid_substation_file = os.path.join(inputFolder, 'PID_Attributes', 'substation.csv')
pid_substation_df = pd.read_csv(pid_substation_file)

# Set file path and read csv for potential capacity for wind
cap_w_path = os.path.join(inputFolder, 'Potential_Installed_Capacity', 'wind_land_capacity.csv')
cap_w_df = pd.read_csv(cap_w_path)

# Set file path and read csv for associated GEA
pid_gea_file = os.path.join(inputFolder, 'PID_Attributes', 'GEAs.csv')
pid_gea_df = pd.read_csv(pid_gea_file)

''' ============================
Define optimization function given a single value
============================ '''

def runWindCost(PID, output_df_arg):

    print('PID: ', PID)

    ''' ============================
    Set scalar parameters
    ============================ '''

    # CAPITAL AND OM COSTS
    if ATBreleaseYr_scen == "2022" and ATBcapexYr_scen == "2025":
        # Define capital expenditure for wind and solar (in USD/MW) 
        # 2022 ATB advanced scenario
        capEx_w = 1081*1000 # class 5, 2025
        capEx_s = 922*1000 # class 5, 2025
        # OPERATION & MAINTENANCE COSTS
        om_w = 39*1000 # class 5, 2025
        om_s = 17*1000 # class 5, 2025
            

    if ATBreleaseYr_scen == "2022" and ATBcapexYr_scen == "2030":
        # Define capital expenditure for wind and solar (in USD/MW) 
        # 2022 ATB advanced scenario
        capEx_w = 704*1000 # class 5, 2030
        capEx_s = 620*1000 # class 5, 2030
        # OPERATION & MAINTENANCE COSTS
        om_w = 34*1000 # class 5, 2030
        om_s = 13*1000 # class 5, 2030
    
        
    if ATBreleaseYr_scen == "2023" and ATBcapexYr_scen == "2025":
        # 2023 ATB advanced scenario
        capEx_w = 1244*1000 # class 5, 2025
        capEx_s = 1202*1000 # class 5, 2025
        om_w = 27*1000 # class 5, 2025
        om_s = 20*1000 # class 5, 2025
        
    
    if ATBreleaseYr_scen == "2023" and ATBcapexYr_scen == "2030":
        # 2023 ATB advanced scenario
        capEx_w = 1096*1000 # class 5, 2030
        capEx_s = 917*1000 # class 5, 2030
        om_w = 24*1000 # class 5, 2030
        om_s = 16*1000 # class 5, 2030
        

    # CAPITAL RECOVERY FACTOR
    # Define discount rate for capital recovery factor
    d = 0.04 
    # Define number of years for capital recovery factor
    n = 25
    n_bat = 12.5 # life of battery
    # Define capital recovery factor (scalar)
    CRF = (d*(1+d)**n)/((1+d)**n - 1)
    CRFbat = (d*(1+d)**n_bat)/((1+d)**n_bat - 1)
    ## denominator of battery 
    denom_batt = (1 + d)**n_bat

    # TRANSMISSION AND SUBSTATION COSTS
    # Define USD2018 per km per MW for total transmission costs per MW
    # USD2018 per km for $500 MW of capacity; divide by 500 to obtain per MW value
    i = 572843/500
    # Define per MW for total transmission costs per MW
    # per 200 MW; divide by 200 to obtain per MW value
    sub = 7609776/200
    # Define kilometers for total transmission costs per MW
    # Corresponding distance to closest substation 115kV or higher by PID
    km = pid_substation_df.loc[pid_substation_df['PID'] == PID, 'distance_km'].values[0]
    # Define total transmission costs per MW (USD per MW cost of tx + substation)
    totalTx_perMW = i*km+sub

    # SOLAR AND WIND POTENTIAL INSTALLED CAPACITY
    # Define potential installed capacity
    cap_w = cap_w_df.loc[cap_w_df['PID'] == PID, 'p_cap_mw'].iloc[0]

    # TRANSMISSION CAPACITY
    # Define associated transmission substations capacity in MW
    ## size to wind capacity * certain percentage
    tx_MW = cap_w * tx_scen

    ''' ============================
    Import data
    ============================ '''

    ## WHOLESALE ELECTRICITY PRICES
    # Determine GEA associated with PID
    gea = pid_gea_df.loc[pid_gea_df['PID'] == PID, 'gea'].values[0]
    
    # Set filepath where wholesale electricity prices are for each GEA
    ePrice_df_folder = os.path.join(inputFolder, cambium_scen + cambium_scen_yr_append, PTC_scen)
    ePrice_path = os.path.join(ePrice_df_folder, f'cambiumHourly_{gea}.csv')
    ePrice_df_wind = pd.read_csv(ePrice_path)
    ePrice_df_wind.drop("hour", axis = 1, inplace = True)

    ## WIND CAPACITY FACTORS
    cf_w_path = os.path.join(inputFolder, 'SAM', 'Wind_Capacity_Factors', f'capacity_factor_PID{PID}.csv')
    cf_w_df = pd.read_csv(cf_w_path)
    #cf_w_df.drop("hour", axis = 1, inplace = True)
    
    ''' ============================
    Calculate revenue, costs, and profit
    ============================ '''

    ## Revenue = sum(ePrice*CF*windCap)
    revenue = (ePrice_df_wind*cf_w_df*cap_w).sum().sum()
    
    ## NPV of lifetime costs = capex + NPV of OM + capex of tx
    costs = cap_w*capEx_w + ((cap_w*om_w)/CRF) + (tx_MW*totalTx_perMW) 
    
    profit = revenue - costs
    
    # === CALCULATE POTENTIAL GENERATION FOR WIND
    ## potential generation  = capacity_wind * CF_hour
    potentialGen_wind = pd.melt(cf_w_df, value_vars = [str(item) for item in range(1,26)], id_vars = "hour", var_name = "year")
    potentialGen_wind["potentialGen"] = cap_w * potentialGen_wind['value']
    potentialGen_wind_annualSum = potentialGen_wind[["potentialGen", "year"]].groupby("year").sum()
    
    ## discount annual revenue and calculate NPV
    potentialGen_wind_annualSum = potentialGen_wind_annualSum.reset_index()
    potentialGen_wind_annualSum['year'] = potentialGen_wind_annualSum["year"].astype('int')
    
    potentialGen_wind_annualSum.sort_values(by = ['year'], inplace = True)
    potentialGen_wind_annualSum['potentialGen_discounted'] = potentialGen_wind_annualSum['potentialGen'] / (1+d)**potentialGen_wind_annualSum['year']
    potentialGen_wind_lifetime_discounted = potentialGen_wind_annualSum['potentialGen_discounted'].sum()
    potentialGen_wind_lifetime = potentialGen_wind_annualSum['potentialGen'].sum()
    
    LCOE = costs/potentialGen_wind_lifetime_discounted
    LVOE = revenue/potentialGen_wind_lifetime_discounted
    NVOE = profit/potentialGen_wind_lifetime_discounted
    
    # append rows to output_df
    output_df = output_df_arg.append({'PID' : int(PID), 
                    'solar_capacity' : "NA", 
                    'wind_capacity' : cap_w, 
                    'solar_wind_ratio' : "NA", 
                    'tx_capacity' : tx_MW, 
                    'batteryCap' : "NA", 
                    'batteryEnergy': "NA", 
                    'revenue': revenue, 
                    'cost': costs, 
                    'profit': profit,
                    'LCOE': LCOE,
                    'LVOE': LVOE,
                    'NVOE': NVOE,
                    'potentialGen_wind_lifetime': potentialGen_wind_lifetime,
                    'potentialGen_solar_lifetime': "NA",
                    'solar_wind_ratio_potentialGen': "NA", 
                    "actualGen_lifetime": "NA",
                    "potentialGen_lifetime": "NA",
                    "exportGen_lifetime": "NA",
                    "curtailment": "NA"}, ignore_index = True)
    
    return output_df


''' ============================
Define optimization function given a list 
============================ '''

def runOptimizationLoop(PID_list, output_df_arg):
    i = 0 
    for PID in PID_list:
        while True:
            try: 
                i = i + 1
                #print(i)
                output_df_arg = runWindCost(PID, output_df_arg)
                output_df_arg.to_csv(output_df_path_csv, index = False)
            except Exception as exc:
                print(exc)
                # Save df to csv
                return output_df_arg
                output_df_arg.to_csv(output_df_path_csv, index = False)
            break  
    
    return output_df_arg

''' ============================
Execute loop
============================ '''

start_time = time.time()

output_df_complete = runOptimizationLoop(PIDsList_needToRun, output_df)  

print("**** Completed", scenario_filename_combined)

end_time = time.time()

print('Time taken:', end_time - start_time, 'seconds')