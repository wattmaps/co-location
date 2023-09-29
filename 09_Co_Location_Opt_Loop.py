''' ============================
Import packages and set directory
============================ '''

import os, sys
import time
import pandas as pd
import numpy as np
import shutil
from pyomo.environ import *
from pyomo.opt import SolverFactory
import cplex

start_time = time.time()

current_dir = os.getcwd()
print(current_dir)

inputFolder = os.path.join(current_dir, 'data')

''' ============================
Set solver
============================ '''

solver = 'cplex'

if solver == 'cplex':
    opt= SolverFactory('cplex', executable = '/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex')
    opt.options['mipgap'] = 0.005

''' ============================
Define functions to generate dictionary objects from dfs
============================ '''

# FUNCTION 1
# Create dictionary from ix1 vector, 
# where i_indexName is name of index as a string, 
# and i_indexList is a list of those index names

def pyomoInput_dfVectorToDict(df, i_indexName, i_indexList):
    # Merge index list to df of vector data
    df_merged = pd.concat([pd.DataFrame({i_indexName: i_indexList}), df], axis = 1)
    
    # Set index
    df_merged.set_index(i_indexName, inplace = True)
    return df_merged.to_dict()

# FUNCTION 2
# Create dictionary from matrix 
def pyomoInput_matrixToDict(df, i_indexName, j_indexNames_list):

    # Melt by indicies
    df_melt = pd.melt(df, id_vars = i_indexName, value_vars = j_indexNames_list)
    
    # Melt all j_indexNames_list (columns) into single 'variable' column
    # Single 'variable' column is second index
    df_melt.set_index([i_indexName, 'variable'], inplace = True)
    out_df = df_melt.to_dict()['value']
    return out_df


''' ============================
Initialize df and filepath
============================ '''

# Create a df with column names
output_df = pd.DataFrame(columns = ['PID', 'solar_capacity', 'wind_capacity', 'solar_wind_ratio', 'revenue', 'cost', 'profit']) #tx_capacity

# Create sequence of PIDs (n=1335) and add to PID column
seq = list(range(1, 1336))
output_df['PID'] = seq

# Save df to csv 
# output_df.to_csv(os.path.join(inputFolder, 'model_results.csv'), index = False)

output_df_path = os.path.join(inputFolder, 'model_results_test.csv')
# output_df = pd.read_csv(output_df_path, engine = 'python')


''' ============================
Define optimization function given a single value
============================ '''

def runOptimization(PID):

    print('PID: ', PID)

    ''' ============================
    Set scalar parameters
    ============================ '''

    # CAPITAL AND OPERATION & MAINTENANCE COSTS
    # Define capital expenditure for wind and solar (in USD/MW) ## updated using 2023 ATB and 2025 moderate scenario
    capEx_w = 1268*1000 
    capEx_s = 1248*1000
    # Define operations & maintenance costs for wind and solar (in USD/MW/yr)
    om_w = 28.8*1000 
    om_s = 20.5*1000 

    # CAPITAL RECOVERY FACTOR
    # Define discount rate for capital recovery factor
    d = 0.04 
    # Define number of years for capital recovery factor
    n = 25
    # Define capital recovery factor (scalar)
    CRF = (d*(1+d)**n)/((1+d)**n - 1)

    # TRANSMISSION AND SUBSTATION COSTS
    # Define USD2018 per km per MW for total transmission costs per MW
    # USD2018 per km for $500 MW of capacity; divide by 500 to obtain per MW value
    i = 572843/500
    # Define per MW for total transmission costs per MW
    # per 200 MW; divide by 200 to obtain per MW value
    sub = 7609776/200
    # Define kilometers for total transmission costs per MW
    # Corresponding distance to closest substation 115kV or higher by PID
    pid_substation_file = os.path.join(inputFolder, 'PID_Attributes', 'substation.csv')
    pid_substation_df = pd.read_csv(pid_substation_file)
    km = pid_substation_df.loc[pid_substation_df['PID'] == PID, 'distance_km'].values[0]
    # Define total transmission costs per MW (USD per MW cost of tx + substation)
    totalTx_perMW = i*km+sub

    # SOLAR AND WIND POTENTIAL INSTALLED CAPACITY
    # Set filepath and read csv for potential capacity for solar
    cap_s_path = os.path.join(inputFolder, 'Potential_Installed_Capacity', 'solar_land_capacity.csv')
    cap_s_df = pd.read_csv(cap_s_path)
    cap_s = cap_s_df.loc[cap_s_df['PID'] == PID, 'solar_installed_cap_mw'].iloc[0]

    # Set file path and read csv for potential capacity for wind
    cap_w_path = os.path.join(inputFolder, 'Potential_Installed_Capacity', 'wind_land_capacity.csv')
    cap_w_df = pd.read_csv(cap_w_path)
    cap_w = cap_w_df.loc[cap_w_df['PID'] == PID, 'p_cap_mw'].iloc[0]

    # TRANSMISSION CAPACITY
    # Define associated transmission substations capacity in MW
    if cap_w <= 100:
        tx_MW = cap_w 
    else:
        tx_MW = pid_substation_df.loc[pid_substation_df['PID'] == PID, 'substation_mw'].values[0]

    ''' ============================
    Set vector parameters
    ============================ '''

    ## WHOLESALE ELECTRICITY PRICES
    pid_gea_file = os.path.join(inputFolder, 'PID_Attributes', 'GEAs.csv')
    pid_gea_df = pd.read_csv(pid_gea_file)

    # Determine GEA associated with PID
    gea = pid_gea_df.loc[pid_gea_df['PID'] == PID, 'gea'].values[0]

    # Set filepath where wholesale electricity prices are for each GEA
    ePrice_df_folder = os.path.join(inputFolder, 'Cambium22_Electrification', 'Cash_Flow')
    ePrice_path = os.path.join(ePrice_df_folder, f'cambiumHourly_{gea}.csv')
    ePrice_df_wind = pd.read_csv(ePrice_path)

    ## SOLAR AND WIND CAPACITY FACTORS
    cf_s_path = os.path.join(inputFolder, 'SAM', 'Solar_Capacity_Factors', f'capacity_factor_PID{PID}.csv')
    # cf_s_path = inputFolder + '/PID1_CF_Wide_Matrix' + '/SOLAR_capacity_factor_PID' + str(PID) + '.csv'
    cf_s_df = pd.read_csv(cf_s_path)
    ## subset to only the capacity factor column
    # cf_only_s_df = cf_s_df.loc[:,'energy_generation']

    cf_w_path = os.path.join(inputFolder, 'SAM', 'Wind_Capacity_Factors', f'capacity_factor_PID{PID}.csv')
    # cf_w_path = inputFolder + '/PID1_CF_Wide_Matrix' + '/WIND_capacity_factor_PID' + str(PID) + '.csv'
    cf_w_df = pd.read_csv(cf_w_path)
    ## subset to only the capacity factor column
    # cf_only_w_df = cf_w_df.loc[:,'energy_generation']
    
    ''' ============================
    Initialize model
    ============================ '''

    model = AbstractModel()

    ''' ============================
    Set hour and year indices
    ============================ '''

    # Extract hour index
    hour = cf_s_df.loc[:,'hour'].tolist()

    # Add hour list to model
    model.t = Set(initialize = hour)
    
    # Set year index
    year = list(range(1, 26))
    year_char = [str(n) for n in year]
    
    # Add year list to model
    model.y = Set(initialize = year_char)
    
    # Add coupled index to model
    model.HOURYEAR = model.t * model.y

    ''' ============================
    Set vector parameters as dictionaries
    ============================ '''

    ## ELECTRICITY PRICES ---
    # Electricity prices for wind and solar
    # ePrice_wind_hourly = next(iter(pyomoInput_dfVectorToDict(ePrice_df_wind, 'hour', hour).values()))
    # ePrice_solar_hourly = next(iter(pyomoInput_dfVectorToDict(ePrice_df_wind, 'hour', hour).values()))

    ## Adapted from wind_zones_v2 script
    ePrice_wind_hourly = pyomoInput_matrixToDict(ePrice_df_wind, 'hour', year_char)
    #ePrice_solar_hourly = next(iter(pyomoInput_matrixToDict(ePrice_df_wind, 'hour', year)))

    ePrice_wind_hourly[1, str(year[0])]

    # Set parameter
    # model.eprice_wind = Param(model.t, default = ePrice_wind_hourly) # price of wind at each hour
    # model.eprice_solar = Param(model.t, default = ePrice_solar_hourly) # price of solar at each hour

    ## Adapted from wind_zones_v2 script
    model.eprice_wind = Param(model.HOURYEAR, default = ePrice_wind_hourly) # price of wind at each hour
    #model.eprice_solar = Param(model.HOURYEAR, default = ePrice_solar_hourly) # price of solar at each hour

    ## CAPACITY FACTORS ---
    # Extract wind capacity factor as vector
    # wind_cf_hourly = next(iter(pyomoInput_dfVectorToDict(cf_only_w_df, 'hour', hour).values()))
    # Set parameter
    # model.cf_wind = Param(model.t, default = wind_cf_hourly)

    # Extract solar capacity factor as vector
    # solar_cf_hourly = next(iter(pyomoInput_dfVectorToDict(cf_only_s_df, 'hour', hour).values()))
    # Set parameter
    # model.cf_solar = Param(model.t, default = solar_cf_hourly)

    ## Adapted from wind_zones_v2 script
    wind_cf_hourly = pyomoInput_matrixToDict(cf_w_df, 'hour', year_char)
    solar_cf_hourly = pyomoInput_matrixToDict(cf_s_df, 'hour', year_char)
    #solar_cf_hourly = next(iter(pyomoInput_matrixToDict(cf_s_df, 'csv', 'hour', year)))

    model.cf_wind = Param(model.HOURYEAR, default = wind_cf_hourly)
    model.cf_solar = Param(model.HOURYEAR, default = solar_cf_hourly)


    ''' ============================
    Set scalar parameters
    ============================ '''

    model.capEx_w = Param(default = capEx_w)
    model.capEx_s = Param(default = capEx_s)
    model.om_w = Param(default = om_w)
    model.om_s = Param(default = om_s)
    model.n = Param(default = n)
    model.CRF = Param(default = CRF)
    model.capEx_tx = Param(default = totalTx_perMW)
    model.pot_w = Param(default = cap_w)
    model.pot_s = Param(default = cap_s)
    model.tx_capacity = Param(default = tx_MW)
    # model.batt_rtEff_sqrt = Param(default = rtEfficiency_sqrt)
    # model.batt_power_cost = Param(default = battPowerCost)
    # model.batt_energy_cost = Param(default = battEnergyCost)

    ''' ============================
    Set decision, slack, and battery variables
    ============================ '''

    # DECISION VARIABLES ---
    model.solar_capacity = Var(within = NonNegativeReals)
    model.wind_capacity = Var(within = NonNegativeReals) 
    # model.tx_capacity = Var(within = NonNegativeReals)

    # SLACK VARIABLES ---
    # Slack variable potential generation (without considering curtailment)
    # model.potentialGen = Var(model.HOURYEAR)
    model.potentialGen = Var(model.HOURYEAR)
    # Slack variable actual generation (considering curtailment)
    # model.actualGen = Var(model.HOURYEAR)
    model.actualGen = Var(model.HOURYEAR)  
    # Slack variable for lifetime revenue
    model.revenue = Var()
    # Slack variable for lifetime costs
    model.cost = Var()
    
    # BATTERY VARIABLES ---
    ## Maximum energy storage of battery
    #model.duration_batt = Var(within=NonNegativeReals)
    ## Maximum power of battery
    #model.P_batt_max = Var(within=NonNegativeReals)
    ## Maximum energy of battery
    #model.E_batt_max = Var(within=NonNegativeReals)
    ## charging power in time t
    #model.P_char_t = Var(model.HOURYEAR)
    ## discharging power in time t
    #model.P_dischar_t = Var(model.HOURYEAR)
    ## Energy of battery in time t
    #model.E_batt_t = Var(model.HOURYEAR)
    ## Losses while charging in time t
    #model.L_char_t = Var(model.HOURYEAR)
    ## Losses while discharging in time t
    #model.L_dischar_t = Var(model.HOURYEAR)
    ## Export of electricity to grid in time t
    #model.Export_t = Var(model.HOURYEAR)

    ''' ============================
    Define objective function
    ============================ '''
    
    # Maximize profit
    def obj_rule(model):
        return model.revenue - model.cost
    model.obj = Objective(rule = obj_rule, sense = maximize)

    ''' ============================
    Define constraints
    ============================ '''

    ## Constraint (1) ---
    ## Define potential generation = CF*capacity for wind and solar
    # combines both forms of generation
    def potentialGeneration_rule(model, t, y):
        return model.potentialGen[t, y] == model.cf_solar[t, y]*model.solar_capacity + model.cf_wind[t, y] * model.wind_capacity
    model.potentialGeneration = Constraint(model.HOURYEAR, rule=potentialGeneration_rule)

    ## Constraint (2) ---
    ## Define actual generation is less than or equal to potential generation
    # how much is actually being delivered to the grid, less than the tx constraint
    def actualGenLTEpotentialGen_rule(model, t, y):
        return model.actualGen[t,y] <= model.potentialGen[t,y]
    model.actualGenLTEpotentialGen = Constraint(model.HOURYEAR, rule = actualGenLTEpotentialGen_rule)

    ## Constraint (3) ---
    ## Define actual generation must be less than or equal to transmission capacity
    def actualGenLTEtxCapacity_rule(model, t, y):
        return model.actualGen[t, y] <= model.tx_capacity
    model.actualGenLTEtxCapacity = Constraint(model.HOURYEAR, rule = actualGenLTEtxCapacity_rule)

    ## Constraint (4) ---
    ## Define lifetime costs (equation #2) in net present value = overnight capital costs + NPV of fixed O&M (using annualPayments = CRF*NPV)
    def lifetimeCosts_rule(model):
        return model.cost == (model.solar_capacity*model.capEx_s) + ((model.solar_capacity*model.om_s)/model.CRF) + \
            (model.wind_capacity*model.capEx_w) + ((model.wind_capacity*model.om_w)/model.CRF) + \
                (model.tx_capacity*model.capEx_tx) # + \
                # model.P_batt_max * model.batt_power_cost + model.E_batt_max * model.batt_energy_cost
    model.annualCosts = Constraint(rule = lifetimeCosts_rule)

    ## Constraint (5) ---
    ## Ensure that capacity is less than or equal to potential for both wind and solar (equation #5)

    # Define rule for solar
    def max_capacity_solar_rule(model):
        return model.solar_capacity <= model.pot_s
    model.maxCapacity_solar = Constraint(rule=max_capacity_solar_rule)

    # Define rule for wind
    def max_capacity_wind_rule(model):
        return model.wind_capacity <= model.pot_w
    model.maxCapacity_wind = Constraint(rule=max_capacity_wind_rule)

    ## Constraint (6) ---
    ## Define lifetime revenue
    def lifetimeRevenue_rule(model):
        return model.revenue == sum(sum(model.actualGen[t, y] * model.eprice_wind[t, y] for t in model.t) for y in model.y)
    model.lifetimeRevenue = Constraint(rule = lifetimeRevenue_rule)

    ## Constraint (7) ---
    ## Check that transmission capacity is less than wind capacity 
    # will always size tx capacity to wind capacity, never undersizes so could change tx_capacity == wind_capacity
    #def tx_wind_rule(model):
        #return model.tx_capacity <= model.wind_capacity #+ model.solar_capacity
    #model.tx_wind = Constraint(rule=tx_wind_rule)


    ## CONSTRAINT (1 - BATTERY) ---
    ## Battery cannot simultaneously charge and discharge
   # def batt_noSimultaneousChargingDischarging_rule(model, t, y):
     #   return model.P_char_t[t,y] * model.P_dischar_t[t,y] == 0
   # model.batt_noSimultaneousChargingDischarging = Constraint(model.HOURYEAR, rule = batt_noSimultaneousChargingDischarging_rule)
    
    ## CONSTRAINT (2 - BATTERY) --- CHECK THIS!
    ## No charging in hour 1
    #def batt_nochargingAtTime1_rule(model, t, y):
      #  return model.P_char_t[1] - model.P_dischar_t[1] == 0
    # model.batt_nochargingAtTime1 = Constraint(model.HOURYEAR, rule = batt_nochargingAtTime1_rule)
    
    ## CONSTRAINT (3 - BATTERY) --- 
    ## initiate the battery charge at time  = 0 at 50% of maximum energy storage 
   # def batt_startAt50percent_rule(model, t, y):
       # return model.E_batt_t[1] == 0.5 * model.E_batt_max
   # model.batt_startAt50percent = Constraint(model.HOURYEAR, rule = batt_startAt50percent_rule)
    
    ## CONSTRAINT (4 - BATTERY) --- 
    ## the losses while charging in hour t is equal to the charging power times 1 - the square root of the round trip efficiency
   # def batt_loss_charging_rule(model, t, y):
        # return model.L_char_t[t, y] == model.P_char_t[t, y] * (1 - model.batt_rtEff_sqrt)
    # model.batt_loss_charging = Constraint(model.HOURYEAR, rule = batt_loss_charging_rule)
    
    ## CONSTRAINT (5 - BATTERY) --- 
    ## the losses while discharging in hour t is equal to the discharging power plus the losses while discharging times 1 - the square root of the round trip efficiency
    # def batt_loss_discharging_rule(model, t, y):
        # return model.L_dischar_t[t, y] == (model.P_dischar_t[t, y] + model.L_dischar_t[t, y]) * (1 - model.batt_rtEff_sqrt)
   #  model.batt_loss_discharging = Constraint(model.HOURYEAR, rule = batt_loss_discharging_rule)

    ## CONSTRAINT (6 - BATTERY) --- 
    ## energy balance of the battery is equal to the energy in the previous hour plus the charging power in hour t minus discharging power minus losses
    # def batt_energyBalance_rule(model, t, y):
       # return model.E_batt_t[t, y] == model.E_batt_t[t-1, y] + model.P_char_t[t, y] - model.P_dischar_t[t, y] - model.L_char_t[t, y] - model.L_dischar_t[t, y]
   # model.batt_energyBalance = Constraint(model.HOURYEAR, rule = batt_energyBalance_rule)
    
    ## CONSTRAINT (7 - BATTERY) --- 
    ## Charge in hour t must be less than or equal to the amt of potential generation 
    # def batt_chargeLessThanGeneration_rule(model, t, y):
        # return -model.P_char_t[t, y] == model.potentialGen[t, y]
    # model.batt_chargeLessThanGeneration = Constraint(model.HOURYEAR, rule = batt_chargeLessThanGeneration_rule)
    
    ## CONSTRAINT (8 - BATTERY) ---  
    ## Charge in hour t must be less than or equal to the amt of potential generation 
    # def batt_dischargeLessThanPowerCapacity_rule(model, t, y):
        # return model.P_dischar_t[t, y] <= model.P_batt_max
    # model.batt_dischargeLessThanPowerCapacity = Constraint(model.HOURYEAR, rule = batt_dischargeLessThanPowerCapacity_rule)
    
    ## CONSTRAINT (9 - BATTERY) --- 
    ## Electricity exported to the grid is equal to actual generation plus the battery dicharge minus the battery charging power
    #def batt_export_rule(model, t, y):
        #return model.Export_t[t, y] == model.actualGen[t, y] + model.P_dischar_t[t, y] - model.P_char_t[t, y]
    #model.batt_export = Constraint(model.HOURYEAR, rule = batt_export_rule)
    
    ## CONSTRAINT (10 - BATTERY) --- 
    ## Energy in battery at time t must be less than or equal to 
   # def batt_export_rule(model, t, y):
    #    return model.Export_t[t, y] == model.actualGen[t, y] + model.P_dischar_t[t,y] - model.P_char_t[t,y]
   # model.batt_chargeLessThanGeneration = Constraint(model.HOURYEAR, rule = batt_chargeLessThanGeneration_rule)
    
    
    ## CONSTRAINT (12 - BATTERY) --- 
    ## Maximum battery energy is equal to the battery duration times the maximum power of the battery
  #  def batt_maxEnergy_rule(model):
       # return model.E_batt_max = model.P_batt_max * model.duration_batt
    # model.batt_maxEnergy = Constraint(rule = batt_maxEnergy_rule)


    ''' ============================
    Execute optimization
    ============================ '''
    model_instance = model.create_instance()
    results = opt.solve(model_instance, tee = True)

    # Store variable values from optimization
    solar_capacity = model_instance.solar_capacity.value
    wind_capacity = model_instance.wind_capacity.value
    # tx_capacity = model_instance.tx_capacity.value
    revenue = model_instance.revenue.value
    cost = model_instance.cost.value
    profit = model_instance.obj()
    solar_wind_ratio = solar_capacity/wind_capacity

    output_df.loc[output_df['PID']== PID] = [PID, solar_capacity, wind_capacity, solar_wind_ratio, revenue, cost, profit] # tx_capacity

''' ============================
Define optimization function given a list 
============================ '''

def runOptimizationLoop(PID_list):
    i = 0 
    for PID in PID_list:
        while True:
            try: 
                i = i + 1
                #print(i)
                runOptimization(PID)
                if i == 300:
                    output_df.to_csv(output_df_path, index = False)
                    print('Saved to file')
                    i = 0
            except Exception as exc:
                print(exc)
                # Save df to csv
                output_df.to_csv(output_df_path, index = False)
                # PAUSE
                # time.sleep(5)
                # continue
            break  
        
    # Save df to csv
    output_df.to_csv(output_df_path, index = False)

''' ============================
Execute optimization loop
============================ '''

PID_start = 1
PID_end = PID_start + 1
start_time = time.time()
PID_list_in = list(range(PID_start, PID_end, 1))
runOptimizationLoop(PID_list_in)  
end_time = time.time()
print('Time taken:', end_time - start_time, 'seconds')  